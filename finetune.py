# ============================================================
# finetune.py — LLM-Mini Fine-tuning (4 Personalities)
# ============================================================
# Usage:
#   python finetune.py --task story
#   python finetune.py --task poetry
#   python finetune.py --task farmer
#   python finetune.py --task qa
# ============================================================

import os
import sys
import math
import time
import json
import random
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import transformers
transformers.logging.set_verbosity_error()

import config
from model.gpt import build_model


# ============================================================
# DATASET
# ============================================================

class FinetuneDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.samples   = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line)["text"])
        print(f"[Dataset] Loaded {len(self.samples):,} from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(
            self.samples[idx],
            truncation=True,
            max_length=config.MAX_SEQ_LEN + 1
        )
        if len(ids) < config.MAX_SEQ_LEN + 1:
            ids += [config.EOS_TOKEN_ID] * (config.MAX_SEQ_LEN + 1 - len(ids))
        ids = torch.tensor(ids[:config.MAX_SEQ_LEN + 1], dtype=torch.long)
        return ids[:-1], ids[1:]


# ============================================================
# DATA BUILDERS
# ============================================================

def build_story_data(tokenizer):
    """TinyStories — 3,000 real diverse short stories."""
    print("[Data] Streaming TinyStories (3,000 samples)...")
    from datasets import load_dataset

    cfg     = config.PROMPTS["story"]
    samples = []

    try:
        ds = load_dataset("roneneldan/TinyStories",
                          split="train", streaming=True)
        for item in ds:
            if len(samples) >= 3000:
                break
            text = item.get("text", "").strip()
            if len(text) < 150 or len(text) > 1500:
                continue
            first = text.split(".")[0].strip()[:80]
            full  = (cfg["prefix"] + first +
                     cfg["response"] + text + cfg["end"])
            samples.append({"text": full})
    except Exception as e:
        print(f"[Data] TinyStories failed: {e}, using manual stories...")
        samples = _manual_stories(cfg)

    print(f"[Data] Story samples: {len(samples):,}")
    return samples


def _manual_stories(cfg):
    stories = [
        ("a brave little girl named Priya",
         "Once upon a time in a small village near the mountains lived a brave little girl named Priya. Every morning she climbed the hills to collect wildflowers for her mother. One day she heard a soft crying sound from behind a large rock. She found a baby deer with its leg caught between two stones. Without hesitating, Priya carefully freed the deer and gently stroked its head until it stopped trembling. The deer looked at her with big brown eyes and then bounded away into the forest. That evening her grandmother said she had the heart of a lion."),
        ("a clever fox named Rusty",
         "Deep in the green forest lived a clever fox named Rusty who was known for solving problems. When the river flooded and blocked the path to the berry bushes all the animals were worried about the coming winter. Rusty studied the river carefully for three days. Then he showed everyone how to stack fallen logs and flat stones to make a narrow bridge. One by one the animals crossed safely. That winter nobody went hungry and every animal in the forest brought Rusty a gift of food to say thank you."),
        ("a boy who loved stars",
         "Arjun was a young boy who stayed awake every night watching the stars from his rooftop. His grandmother sat beside him telling stories about each constellation. One clear night Arjun noticed a small light moving slowly across the sky that was different from the others. He drew its exact position every night for two weeks in his notebook. His science teacher saw the drawings and was so amazed that she sent them to a university observatory. The astronomers confirmed that Arjun had tracked a newly discovered asteroid. He became the youngest person in his city to contribute to astronomy."),
        ("a magical garden behind the school",
         "Behind the old school in Riverside town there was a forgotten garden that nobody visited for years. Weeds grew taller than the fence and old broken pots lay scattered everywhere. A quiet girl named Meera decided one summer to clean it up alone. Every morning before school she pulled weeds and watered seeds she had saved from her mother's kitchen. By October the garden was full of sunflowers marigolds and tomatoes. Children who had never spoken to Meera before came to help and became her closest friends. The garden became the most visited place in the whole town."),
        ("the little lighthouse keeper",
         "On a rocky island far from the shore lived old Thomas the lighthouse keeper and his granddaughter Lily. Every night Thomas climbed the long spiral staircase to light the great lamp that guided ships safely past the dangerous rocks. One stormy night Thomas fell ill and could not climb the stairs. Without the light a ship was heading straight toward the rocks. Nine-year-old Lily had never climbed the stairs alone but she took a deep breath and began climbing one step at a time. She reached the top and lit the lamp just in time. The captain of the ship later came to the island and shook her hand and called her the bravest person he had ever met."),
        ("a kind old man and his dog",
         "In a quiet town at the edge of a forest lived an old man named Samuel and his dog Bruno. Every morning they walked together along the river path and Samuel would stop to feed the birds. One winter morning Bruno suddenly ran off the path and began digging in the snow. Samuel followed and found a small child who had fallen and was covered in snow. Samuel carried the child home and warmed her with tea and blankets. When her parents came they had been searching for hours. Samuel just smiled and said Bruno deserved all the credit."),
        ("the girl who could talk to animals",
         "Maya always knew she was different. When other children played games she sat quietly in the garden listening. She discovered one afternoon that she could understand what the sparrows were saying. They told her which neighbor needed help that day and she would quietly do something kind. She brought flowers to the lonely teacher and left food at the door of the hungry family down the road. Nobody ever knew how she always seemed to know what was needed. Only the sparrows knew her secret."),
        ("the robot who learned kindness",
         "Robot Seven was built to clean the city streets every night. He worked alone while everyone slept. One night he found a small kitten shivering under a bench. His programming said nothing about kittens but something made him stop. He used his heating panel to warm the kitten and stayed beside it until morning. A little girl found them together at sunrise and her face lit up with joy. After that Robot Seven always checked under every bench on his route and the city somehow felt a little warmer."),
    ]
    return [{"text": cfg["prefix"] + t + cfg["response"] + s + cfg["end"]}
            for t, s in stories] * 15


def build_poetry_data(tokenizer):
    """
    merve/poetry with fixes:
    1. NoneType safe handling for poem name field
    2. Skip old English (Renaissance/Medieval) poems
    3. Fall back to manual modern poems if dataset too small
    """
    print("[Data] Loading poetry dataset...")
    from datasets import load_dataset

    cfg     = config.PROMPTS["poetry"]
    samples = []

    # Ages to skip — these produce old English output
    OLD_ENGLISH_AGES = {
        "Renaissance", "Medieval", "Elizabethan",
        "Jacobean", "Augustan", "Restoration"
    }

    try:
        ds = load_dataset("merve/poetry", split="train")
        print(f"[Data] Total poems in dataset: {len(ds)}")

        for item in ds:
            if len(samples) >= 3000:
                break

            content = item.get("content") or ""
            content = content.strip()

            # ── Fix 1: Safe title handling (NoneType fix) ──────
            title = item.get("poem name") or item.get("title") or ""
            title = title.strip() if isinstance(title, str) else ""

            # ── Fix 2: Skip old English poems ──────────────────
            age = item.get("age") or ""
            if any(old in age for old in OLD_ENGLISH_AGES):
                continue

            # ── Fix 3: Quality filters ──────────────────────────
            if not content:
                continue
            if len(content) < 50 or len(content) > 1000:
                continue
            # Skip poems with too many old English words
            old_words = ["doth", "thee", "thou", "thy", "hath",
                         "shalt", "hast", "tis", "twas", "wherefore"]
            if sum(1 for w in old_words if w in content.lower()) > 2:
                continue

            topic = title if title else "life and nature"
            full  = (cfg["prefix"] + topic +
                     cfg["response"] + content + cfg["end"])
            samples.append({"text": full})

        print(f"[Data] Poetry from dataset (modern only): {len(samples):,}")

    except Exception as e:
        print(f"[Data] merve/poetry failed: {e}")

    # Always add manual modern poems as anchor examples
    manual = _manual_poems(cfg)
    samples += manual
    print(f"[Data] Added {len(manual)} manual poems")

    # If still too few samples repeat to get enough training steps
    if len(samples) < 500:
        samples = samples * 5
        print(f"[Data] Repeated to: {len(samples):,}")

    print(f"[Data] Poetry total: {len(samples):,}")
    return samples


def _manual_poems(cfg):
    """
    Hand-written modern English poems.
    Clear rhyme, no old English, good style.
    """
    poems = [
        ("rain and nature",
         "The rain falls soft on trembling leaves,\nAnd whispers secrets to the trees.\nEach drop a story, cold and bright,\nThat washes clean the fading light.\nThe earth drinks deep and flowers grow,\nIn the gentle rain's soft afterglow."),
        ("the passing of time",
         "Time flows like rivers to the sea,\nCarrying all that used to be.\nYesterday a distant shore,\nTomorrow knocking at the door.\nWe stand between the then and now,\nAnd wonder at the why and how."),
        ("friendship",
         "A friend is like the morning sun,\nThat lights your path when day's begun.\nThrough storm and calm they hold your hand,\nThe truest soul in all the land.\nNo mountain high, no valley deep,\nCan break the promises we keep."),
        ("hope",
         "Hope is the bird that sings at dawn,\nBefore the darkness has withdrawn.\nShe perches in the coldest tree,\nAnd sings her song of what could be.\nThough winter comes with frost and snow,\nHer melody keeps hearts aglow."),
        ("the ocean",
         "The ocean speaks in ancient tongue,\nOf battles lost and victories won.\nHer waves the memories of time,\nHer depths a mystery sublime.\nShe calls to those who dare to dream,\nNothing is quite what it may seem."),
        ("moonlight",
         "The moon pours silver on the lake,\nAnd lights the path the dreamers take.\nShe watches over those who sleep,\nAnd guards the promises we keep.\nHer gentle light so calm and still,\nReaches over every hill."),
        ("autumn leaves",
         "The leaves let go without a fight,\nThey spiral down in golden light.\nThe trees stand bare against the sky,\nAnd watch the last warm breezes die.\nYet in this ending there is grace,\nA quiet beauty fills this place."),
        ("a child's wonder",
         "The child looks up at endless sky,\nAnd asks the stars the reason why.\nEach question blooms like morning flower,\nFresh with dew in the early hour.\nIn their eyes the world is new,\nEvery colour every hue."),
        ("mountains",
         "The mountains wear their snow like crowns,\nAnd look with calm on all the towns.\nThey've watched a thousand years go by,\nUnchanging under changing sky.\nTheir silence speaks of things most deep,\nOf ancient promises they keep."),
        ("a letter never sent",
         "I wrote your name in morning frost,\nAnd knew before I wrote it, lost.\nThe words I never found to say,\nMelted with the frost of day.\nBut somewhere in the winter air,\nThose words still float without a care."),
        ("the city at night",
         "The city hums its midnight song,\nA million lives that drift along.\nEach window holds a different light,\nA story hidden in the night.\nThe streets below are cold and bright,\nFull of people out of sight."),
        ("spring morning",
         "The world wakes up in shades of green,\nThe freshest sight I've ever seen.\nThe flowers push through frozen ground,\nAnd birds return without a sound.\nThe air is soft and cool and clear,\nAnd everything feels bright and near."),
        ("the river",
         "The river has no place to go,\nYet still it moves through ice and snow.\nIt carves its path through stone and clay,\nAnd finds the sea without delay.\nIf rivers know no fear of stone,\nThen neither should I face alone."),
        ("an old photograph",
         "I found your face in faded light,\nA photograph from some lost night.\nYour smile was young, your eyes were bright,\nBefore the years stole what was right.\nI set the photograph aside,\nAnd kept the smile somewhere inside."),
        ("the wind",
         "The wind does not remember where it's been,\nIt moves through every place unseen.\nIt touches trees and turns the tide,\nAnd carries seeds from side to side.\nI wish that I could be so free,\nAnd move through life so easily."),
        ("evening",
         "The evening folds the daylight in,\nAnd somewhere new things must begin.\nThe stars come out one by one,\nTo take the places of the sun.\nThe world grows quiet soft and deep,\nAnd draws the tired world to sleep."),
        ("courage",
         "Courage is not the absence of fear,\nIt's taking one more step from here.\nIt's standing up when all seems lost,\nAnd paying whatever is the cost.\nSmall acts of courage change the day,\nAnd light the dark and find the way."),
        ("childhood",
         "We did not know those days would end,\nAround each corner and each bend.\nThe summers stretched like open seas,\nThe winters sparkled through the trees.\nNow looking back I understand,\nHow brief it was, how close at hand."),
        ("gratitude",
         "I am grateful for the morning light,\nFor having made it through the night.\nFor simple food and open sky,\nFor questions left unanswered why.\nFor people kind enough to stay,\nAnd walk beside me on my way."),
        ("the stars",
         "The stars have watched the world below,\nThrough every war and peace we know.\nThey do not judge and do not care,\nThey simply shine through open air.\nAnd when I feel the world too near,\nI look at them and lose my fear."),
    ]
    # Repeat 8x for enough training signal
    return [{"text": cfg["prefix"] + t + cfg["response"] + p + cfg["end"]}
            for t, p in poems] * 8


def build_farmer_data(tokenizer):
    """
    50 hand-written diverse farming examples.
    10 categories × 5 examples each.
    Repeated ×10 = 500 samples, ~100+ training steps.
    """
    print("[Data] Building farmer dataset...")

    cfg = config.PROMPTS["farmer"]

    examples = [
        # ── WHEAT ──────────────────────────────────────────
        ("wheat leaves turning yellow at tips",
         "Yellow leaf tips in wheat indicate potassium deficiency or salt stress. Apply muriate of potash at 25 kg per acre. Avoid waterlogging as it prevents nutrient uptake. If soil is saline apply gypsum at 4 bags per acre. Best time to apply potassium is at tillering stage."),
        ("wheat crop falling down lodging",
         "Lodging in wheat is caused by excess nitrogen, weak stems, or heavy rain. Avoid over-application of urea after tillering. Apply chlormequat chloride growth regulator at 1 litre per acre to strengthen stems. Select lodging-resistant varieties like HD-2967 next season."),
        ("best fertilizer schedule for wheat",
         "Apply full DAP at sowing at 50 kg per acre. Apply first urea dose at 30 kg at crown root initiation around 20-25 days after sowing. Apply second urea dose at 30 kg at tillering stage. Apply potash at 25 kg at jointing for grain filling."),
        ("wheat not germinating after sowing",
         "Poor germination is caused by old seeds, improper sowing depth, or dry soil. Test seed germination rate — it should be above 85 percent. Sow at 4-5 cm depth and irrigate immediately if soil is dry. Treat seeds with thiram fungicide at 2 grams per kg to prevent seed rot."),
        ("when to harvest wheat",
         "Harvest wheat when grain moisture is 12-14 percent and grains are hard and golden yellow. Straw turns yellow and grains cannot be dented by fingernail. Delay causes shattering losses. Harvest in morning to avoid heat losses and dry grain for 2-3 days before storage."),
        # ── RICE ───────────────────────────────────────────
        ("rice leaves have brown spots",
         "Brown spots indicate brown spot disease from Helminthosporium fungus. Apply mancozeb or propiconazole fungicide. Ensure proper potassium nutrition as deficiency increases susceptibility. Remove and burn infected debris after harvest to reduce disease carryover."),
        ("rice tillering is very poor",
         "Poor tillering is caused by nitrogen deficiency, deep water, or poor transplanting. Apply urea at 20 kg per acre at active tillering 15-20 days after transplanting. Maintain water level at 5 cm only. Plant at correct depth of 2-3 cm at 20 by 15 cm spacing."),
        ("rice crop has stunted yellowish growth",
         "Stunted growth in rice usually indicates zinc deficiency which is very common in paddies. Apply zinc sulfate at 25 kg per hectare before transplanting. If symptoms appear after transplanting spray zinc sulfate at 0.5 percent on leaves."),
        ("how to manage water in rice field",
         "Keep saturated soil first week after transplanting. Maintain 5 cm water during active tillering. Drain field for 7-10 days at mid-season for root aeration which saves 30 percent water. Stop irrigation 10 days before harvest for field drying."),
        ("rice has empty grains at harvest",
         "Empty grains are caused by water stress at flowering or boron deficiency. Ensure water supply at flowering stage. Apply urea at panicle initiation. Spray borax at 0.2 percent during booting stage to improve pollination."),
        # ── VEGETABLES ─────────────────────────────────────
        ("tomato leaves are turning yellow",
         "Yellow leaves on tomato indicate nitrogen deficiency or overwatering. Apply nitrogen-rich fertilizer like urea at 1 bag per acre. Reduce watering and ensure proper drainage. Check for fungal infection by looking for spots and apply copper fungicide if found."),
        ("tomato fruit not setting properly",
         "Poor fruit set is caused by high temperature above 35 degrees or boron deficiency. Spray boric acid at 0.2 percent during flowering. Apply potassium fertilizer. Provide shade nets during peak summer and plant heat-tolerant varieties for summer crop."),
        ("onion not forming bulbs",
         "Poor bulb formation is caused by wrong variety for season or excess nitrogen. Use short-day varieties in winter and long-day varieties in spring. Reduce nitrogen after 60 days. Stop irrigation 2 weeks before harvest for bulb maturation."),
        ("brinjal leaves have many holes",
         "Holes in brinjal leaves indicate shoot and fruit borer attack. Install pheromone traps to monitor the pest population. Spray spinosad at 0.3 ml per liter or neem oil at 3 ml per liter. Remove and destroy all bored shoots immediately."),
        ("chilli plants dying suddenly",
         "Sudden death indicates phytophthora root rot. Check stem base for brown water-soaked lesion. Improve field drainage immediately. Apply metalaxyl fungicide drench around plant base and remove affected plants."),
        # ── FRUITS ─────────────────────────────────────────
        ("mango tree leaves have black spots",
         "Black spots indicate anthracnose disease from Colletotrichum fungus. Spray carbendazim or mancozeb every 15 days during humid weather. Prune overcrowded branches for air circulation. Apply copper oxychloride before monsoon as prevention."),
        ("banana plants have yellow leaves",
         "Yellow leaves indicate Panama wilt or nutritional deficiency. For Panama wilt check pseudostem for brown inside and remove infected plant. For nutrition apply urea 200 grams and potassium sulfate 300 grams per plant."),
        ("how to make mango tree flower",
         "Apply paclobutrazol soil drench at 5 grams per meter canopy diameter in September-October. Withhold irrigation for 6-8 weeks before flowering season to induce stress. Spray potassium nitrate at 1 percent to synchronize and enhance flowering."),
        ("citrus leaves have yellow veins",
         "Yellow veins with green leaf tissue indicates iron or manganese deficiency called chlorosis. This is common in alkaline soils. Spray ferrous sulfate at 0.5 percent with citric acid. Apply chelated iron fertilizer for longer effect. Citrus needs soil pH 6.0-7.0."),
        ("papaya plants falling over",
         "Papaya falling is caused by wind, poor root system, or stem rot. Provide bamboo staking support for young plants. Ensure good drainage as waterlogging causes root rot. Apply Trichoderma around stem base to prevent fungal rot."),
        # ── SOIL ───────────────────────────────────────────
        ("soil is very hard and compact",
         "Deep plow with subsoiler to break hardpan at 30-45 cm depth. Add farmyard manure at 10 tons per acre. Practice green manuring with dhaincha. Grow deep-rooted crops like sunflower to break compaction over time."),
        ("soil test shows high pH alkaline",
         "Apply sulfur at 500 kg per hectare to lower pH gradually. Use ammonium sulfate fertilizer instead of urea as it acidifies soil. Apply gypsum at 5 tons per hectare for sodium-affected soils. Regular organic matter addition slowly improves alkaline soil."),
        ("how to improve sandy soil",
         "Add compost at 15-20 tons per hectare every year to improve structure. Apply bentonite clay at 2-3 tons per acre to improve water holding capacity. Use slow-release fertilizers applied in split doses. Drip irrigation is ideal for sandy soils."),
        ("soil has white salt crust on surface",
         "Apply gypsum at 5-10 bags per acre to displace sodium. Provide heavy irrigation to leach salts below root zone and ensure good drainage. Plant salt-tolerant crops like barley or mustard in affected areas."),
        ("which crops grow best in black cotton soil",
         "Black soil is ideal for cotton, soybean, sorghum, chickpea, and wheat. These soils have good water retention and high clay content. Raised bed planting helps manage excess moisture. Avoid growing rice in black soil due to deep cracking."),
        # ── IRRIGATION ─────────────────────────────────────
        ("how to set up drip irrigation",
         "Lay main pipeline along field boundary and connect sub-main lines perpendicular to it. Place drip laterals along crop rows. Install drippers at 30-45 cm intervals for vegetables. Install filter system at water source and flush weekly. Drip saves 40-50 percent water."),
        ("crop wilting even after watering",
         "Wilting after irrigation indicates root rot, compacted soil, or saline water. Check if water drains properly. Test irrigation water EC and do not use above EC of 2. Check roots for brown rot. Improve field leveling for uniform water distribution."),
        ("how much water does mustard need",
         "Apply first irrigation at branching stage 30-35 days after sowing. Second critical irrigation at flowering 55-60 days. Third at pod filling if no rainfall. Total water requirement is 250-350 mm per season. Mustard is moderately drought tolerant."),
        ("borewell water reducing every year",
         "Switch to drip irrigation immediately to save water. Practice rainwater harvesting with farm ponds to collect monsoon runoff. Make percolation pits to recharge groundwater. Grow less water-intensive crops in summer season."),
        ("water not reaching end of field",
         "Field is not level or water pressure is insufficient. Level field using laser leveling — even 5 cm difference causes uneven irrigation. Divide large fields into smaller plots. Irrigate smaller plots sequentially for even water distribution."),
        # ── PESTS ──────────────────────────────────────────
        ("locusts attacking crops",
         "Spray chlorpyrifos or malathion immediately over affected area. Contact agriculture department for aerial spraying if swarm is large. Create noise to drive away swarms before they land. Early morning spraying when locusts are sluggish is most effective."),
        ("termites damaging crop roots",
         "Treat soil with chlorpyrifos at 3 liters per hectare mixed with irrigation water. Apply phorate granules in furrows at sowing. Drench with imidacloprid around plant base for standing crop. Remove tree stumps from field as they attract termites."),
        ("whitefly attack on vegetables",
         "Spray imidacloprid at 0.3 ml per liter or thiamethoxam at 0.2 grams per liter. Use yellow sticky traps at 10 per acre to trap adults. Spray neem oil at 3 ml per liter as organic alternative. Intercrop with marigold to repel whitefly."),
        ("how to do integrated pest management",
         "Monitor fields weekly using sticky traps and visual inspection. Use pest-resistant crop varieties. Release Trichogramma egg parasitoid for bollworm control. Practice crop rotation and intercropping. Use chemical pesticides only when pest crosses economic threshold level."),
        ("rats eating stored grain",
         "Seal all holes in storage room with cement. Place zinc phosphide bait stations outside storage at 10 meter intervals. Use snap traps inside storage. Store grain in metal bins. For large storage use aluminium phosphide fumigation by licensed operator only."),
        # ── FERTILIZER ─────────────────────────────────────
        ("how to use organic manure properly",
         "Apply well-decomposed manure at 10-15 tons per hectare 3-4 weeks before sowing. Never use partially decomposed manure as it burns roots and attracts termites. Vermicompost can be applied at 2-3 tons per hectare. Green manure: sow dhaincha and incorporate at 45 days."),
        ("crop shows nitrogen deficiency signs",
         "Nitrogen deficiency shows as yellowing starting from older lower leaves progressing upward. Plants are stunted with pale stems. Apply urea at 20 kg per acre as top dressing. Spray urea at 2 percent solution on leaves in evening for quick response."),
        ("what is DAP and how to use it",
         "DAP is di-ammonium phosphate containing 18 percent nitrogen and 46 percent phosphorus. Apply at sowing time mixed with soil in furrows at 50 kg per acre. Do not place DAP directly in contact with seeds as it damages germination."),
        ("micronutrient deficiency in crops",
         "Zinc deficiency is most common — apply zinc sulfate at 25 kg per hectare. Boron deficiency causes hollow stem in cauliflower and poor fruit set — spray borax at 0.2 percent. Iron deficiency causes yellowing between veins in alkaline soils — spray ferrous sulfate."),
        ("fertilizer prices are high what to do",
         "Get soil tested and apply only what is needed — most farmers over-apply. Use neem-coated urea which increases nitrogen efficiency by 10-15 percent. Combine organic and chemical fertilizers at half dose each. Use biofertilizers like Rhizobium for pulse crops."),
        # ── HARVEST AND STORAGE ────────────────────────────
        ("how to store onion after harvest",
         "Cure onions in field for 7-10 days after topping to dry outer skin. Store in well-ventilated bamboo or wire mesh racks with good air circulation at 25-30 degrees. Check stored onions weekly and remove any showing rot immediately."),
        ("grain has insects after storage",
         "Clean storage bins thoroughly before filling. Treat storage walls with malathion spray and allow to dry. Mix grain with neem leaves at 1 kg per quintal as natural repellent. Store grain at moisture below 12 percent. Use aluminium phosphide for large storage."),
        ("how to get better price for vegetables",
         "Harvest at proper maturity and grade produce by size and quality. Pack in clean ventilated crates. Form farmer producer groups to negotiate better prices collectively. Check mandi prices online before selling. Use cold storage for extra shelf life."),
        ("wheat grain turning black in storage",
         "Black discoloration indicates Aspergillus or Penicillium mold which produces dangerous aflatoxins. Dry grain below 12 percent moisture before storage. Ensure storage is completely dry with no roof leaks. Never store wet grain. Improve storage ventilation immediately."),
        ("how to reduce losses in tomato after harvest",
         "Harvest at mature green to light red stage for distant markets. Handle very gently to avoid bruising. Pre-cool harvested tomatoes in shade before packing. Use ventilated plastic crates not jute bags. Do not mix mature and immature fruits in same crate."),
        # ── GENERAL ────────────────────────────────────────
        ("how to start organic farming",
         "Start by getting soil tested to understand baseline fertility. Stop chemical fertilizers gradually over 2-3 seasons. Build compost pit using crop residue, kitchen waste, and cow dung. Apply vermicompost and green manure. Get organic certification after 3 years chemical-free farming."),
        ("best crops to grow in summer season",
         "Good summer crops in India are cucumber, bitter gourd, bottle gourd, okra, cluster beans, and cowpea. These crops tolerate heat well. Ensure regular irrigation every 5-7 days in summer. Use mulching to conserve soil moisture. Sow in February-March for best results."),
        ("how to protect crops from frost",
         "Cover young plants with plastic sheets or straw mulch during frost nights. Irrigate fields in the evening before expected frost as moist soil retains heat better. Create windbreaks using trees or nets on the north side. Sprinkle water on plants during frost to create protective ice layer."),
        ("how to test soil at home",
         "Basic home soil test: test pH using pH paper strips available at agri shops. Wet soil with distilled water and dip strip for reading. For texture test: take handful of moist soil and squeeze. Sandy soil falls apart, clay soil forms ribbon, loam holds shape loosely. Send sample to KVK for full nutrient analysis."),
        ("what is crop rotation and why do it",
         "Crop rotation means growing different crops in the same field each season. It prevents buildup of pests and diseases specific to one crop. It improves soil fertility — legumes like chickpea fix nitrogen for the next crop. Example rotation: wheat then rice then mustard then chickpea. Improves yield by 15-20 percent over monoculture."),
    ]

    samples = []
    for situation, advice in examples:
        full = (cfg["prefix"] + situation +
                cfg["response"] + advice + cfg["end"])
        samples.append({"text": full})

    print(f"[Data] Farmer unique examples: {len(samples)}")
    # Repeat ×10 = 500 samples, ~100+ training steps
    return samples * 10


def build_qa_data(tokenizer):
    """SQuAD — 3,000 real diverse Q&A pairs."""
    print("[Data] Streaming SQuAD (3,000 samples)...")
    from datasets import load_dataset

    cfg     = config.PROMPTS["qa"]
    samples = []
    seen    = set()

    try:
        ds = load_dataset("rajpurkar/squad", split="train", streaming=True)
        for item in ds:
            if len(samples) >= 3000:
                break
            question = item.get("question", "").strip()
            answers  = item.get("answers", {})
            ans_list = answers.get("text", []) if answers else []
            answer   = ans_list[0].strip() if ans_list else ""

            if not question or not answer or question in seen:
                continue
            if len(answer) < 5 or len(answer) > 300:
                continue

            seen.add(question)
            full = (cfg["prefix"] + question +
                    cfg["response"] + answer + cfg["end"])
            samples.append({"text": full})
    except Exception as e:
        print(f"[Data] SQuAD failed: {e}")

    print(f"[Data] QA samples: {len(samples):,}")
    return samples


# ============================================================
# TASK REGISTRY
# ============================================================

DATA_BUILDERS = {
    "story":  build_story_data,
    "poetry": build_poetry_data,
    "farmer": build_farmer_data,
    "qa":     build_qa_data,
}

TRAIN_FILES = {
    "story":  config.FINETUNE_STORY_TRAIN,
    "poetry": config.FINETUNE_POETRY_TRAIN,
    "farmer": config.FINETUNE_FARMER_TRAIN,
    "qa":     config.FINETUNE_QA_TRAIN,
}

VAL_FILES = {
    "story":  config.FINETUNE_STORY_VAL,
    "poetry": config.FINETUNE_POETRY_VAL,
    "farmer": config.FINETUNE_FARMER_VAL,
    "qa":     config.FINETUNE_QA_VAL,
}


# ============================================================
# TRAINING UTILITIES
# ============================================================

def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, max_batches=30):
    model.eval()
    total, count = 0.0, 0
    for i, (inp, tgt) in enumerate(val_loader):
        if i >= max_batches:
            break
        inp, tgt = inp.to(config.DEVICE), tgt.to(config.DEVICE)
        with torch.amp.autocast('cuda', enabled=config.USE_FP16):
            _, loss = model(inp, tgt)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def save_checkpoint(model, optimizer, scheduler, scaler,
                    epoch, step, val_losses, path):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch":      epoch,
        "step":       step,
        "model":      raw.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "scaler":     scaler.state_dict(),
        "val_losses": val_losses,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location=config.DEVICE, weights_only=False)
    raw  = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["step"], ckpt["val_losses"]


# ============================================================
# MAIN FINE-TUNE LOOP
# ============================================================

def finetune(task):
    print("=" * 55)
    print(f"  LLM-Mini — Fine-tuning: {task.upper()}")
    print("=" * 55)

    train_path = TRAIN_FILES[task]
    val_path   = VAL_FILES[task]

    # Build data if not already built
    if not os.path.exists(train_path):
        print(f"[Finetune] Building {task} dataset...")
        tokenizer_tmp = GPT2Tokenizer.from_pretrained("gpt2")
        samples = DATA_BUILDERS[task](tokenizer_tmp)

        if not samples:
            print(f"[Finetune] ERROR: No data for {task}")
            sys.exit(1)

        random.shuffle(samples)
        split   = int(len(samples) * 0.95)
        train_s = samples[:split]
        val_s   = samples[split:]

        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        for path, data in [(train_path, train_s), (val_path, val_s)]:
            with open(path, "w", encoding="utf-8") as f:
                for s in data:
                    f.write(json.dumps(s) + "\n")
            print(f"[Finetune] Saved {len(data):,} → {path}")
    else:
        print("[Finetune] Data already exists, skipping build.")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Datasets + loaders
    train_ds     = FinetuneDataset(train_path, tokenizer)
    val_ds       = FinetuneDataset(val_path,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.FINETUNE_BATCH,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.FINETUNE_BATCH,
                              shuffle=False, num_workers=2, pin_memory=True)

    total_steps  = (len(train_loader) // config.FINETUNE_GRAD_ACCUM) \
                   * config.FINETUNE_EPOCHS
    warmup_steps = max(30, total_steps // 10)

    print(f"\n  Train samples  : {len(train_ds):,}")
    print(f"  Val samples    : {len(val_ds):,}")
    print(f"  Steps/epoch    : {len(train_loader)//config.FINETUNE_GRAD_ACCUM:,}")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Warmup steps   : {warmup_steps:,}")
    print()

    # Load pretrained base
    best_pretrain = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    if not os.path.exists(best_pretrain):
        print(f"[Finetune] ERROR: {best_pretrain} not found!")
        print("  Run python train.py first.")
        sys.exit(1)

    print("[Finetune] Loading pretrained weights...")
    model = build_model(config)
    ckpt  = torch.load(best_pretrain, map_location=config.DEVICE,
                       weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.train()

    try:
        model = torch.compile(model)
        print("[Finetune] torch.compile enabled")
    except Exception:
        pass

    decay    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = AdamW([
        {"params": decay,    "weight_decay": config.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.FINETUNE_LR, betas=(0.9, 0.95))

    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    scaler    = torch.amp.GradScaler('cuda', enabled=config.USE_FP16)

    best_val    = float("inf")
    val_losses  = []
    global_step = 0
    start_epoch = 0

    best_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{task}_best.pt")
    last_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{task}_last.pt")

    if os.path.exists(last_ckpt):
        ans = input(f"\n[Resume] Found {task}_last.pt. Resume? (y/n): ")
        if ans.strip().lower() == "y":
            last_epoch, global_step, val_losses = \
                load_checkpoint(last_ckpt, model, optimizer, scheduler, scaler)
            start_epoch = last_epoch + 1
            best_val    = min(val_losses) if val_losses else float("inf")
            print(f"[Resume] Starting epoch {start_epoch+1}, "
                  f"best val: {best_val:.4f}")

    print("\n" + "─" * 55)
    print(f"Starting epoch {start_epoch+1} / {config.FINETUNE_EPOCHS}")
    print("─" * 55 + "\n")

    for epoch in range(start_epoch, config.FINETUNE_EPOCHS):
        epoch_start = time.time()
        epoch_loss  = 0.0
        epoch_steps = 0

        optimizer.zero_grad()

        for step, (inp, tgt) in enumerate(train_loader):
            inp = inp.to(config.DEVICE, non_blocking=True)
            tgt = tgt.to(config.DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=config.USE_FP16):
                _, loss = model(inp, tgt)
                loss    = loss / config.FINETUNE_GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % config.FINETUNE_GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                loss_val     = loss.item() * config.FINETUNE_GRAD_ACCUM
                epoch_loss  += loss_val
                epoch_steps += 1
                global_step += 1

                if global_step % config.LOG_INTERVAL == 0:
                    avg = epoch_loss / epoch_steps
                    lr  = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch+1:02d} | Step {global_step:4d} | "
                          f"Loss {loss_val:.4f} | Avg {avg:.4f} | LR {lr:.2e}")

        avg_train = epoch_loss / max(1, epoch_steps)
        val_loss  = evaluate(model, val_loader)
        val_losses.append(val_loss)

        elapsed = time.time() - epoch_start
        print(f"\n{'═'*55}")
        print(f"  Epoch {epoch+1:02d} | "
              f"Train: {avg_train:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Time: {elapsed/60:.1f} min")
        print(f"{'═'*55}\n")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, global_step, val_losses, best_ckpt)
            print(f"  ⭐ New best! Val Loss: {best_val:.4f}")

        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, global_step, val_losses, last_ckpt)

    print(f"\n🎉 Fine-tuning [{task}] complete!")
    print(f"  Best Val Loss : {best_val:.4f}")
    print(f"  Checkpoint    : {best_ckpt}")
    print(f"\n  Test: python chat.py --task {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=list(DATA_BUILDERS.keys()))
    args = parser.parse_args()
    finetune(args.task)