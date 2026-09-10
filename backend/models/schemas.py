from pydantic import BaseModel, EmailStr


class RegisterReq(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginReq(BaseModel):
    email: EmailStr
    password: str


class TokenReq(BaseModel):
    token: str


class ForgotReq(BaseModel):
    email: EmailStr


class ResetReq(BaseModel):
    token: str
    password: str


class NewChatReq(BaseModel):
    title: str = "New Chat"


class RenameChatReq(BaseModel):
    title: str


class UpdateUsernameReq(BaseModel):
    username: str


class ChangePasswordReq(BaseModel):
    old_password: str
    new_password: str


class ChangeEmailReq(BaseModel):
    new_email: EmailStr
    password: str