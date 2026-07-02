#from passlib.context import CryptContext

#pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory user store — replace with DB in production
#USERS_DB = {
    #"admin": "admin",
    #"vijay": "vijay123",
#}


#def verify_password(plain: str, hashed: str) -> bool:
    #return pwd_context.verify(plain, hashed)


#def authenticate_user(username: str, password: str) -> bool:
    #if username not in USERS_DB:
        #return False
    ##return verify_password(password, USERS_DB[username])


USERS_DB = {
    "admin": "admin123",
    "vijay": "vijay123",
}


def authenticate_user(username: str, password: str) -> bool:
    if username not in USERS_DB:
        return False
    return USERS_DB[username] == password