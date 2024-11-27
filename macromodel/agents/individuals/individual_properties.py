from enum import Enum


class ActivityStatus(Enum):
    EMPLOYED = 1
    UNEMPLOYED = 2
    NOT_ECONOMICALLY_ACTIVE = 3
    FIRM_INVESTOR = 4
    BANK_INVESTOR = 5


class Gender(Enum):
    MALE = 1
    FEMALE = 2


class Education(Enum):
    NONE = 0
    PRIMARY = 1
    LOWER_SECONDARY = 2
    UPPER_SECONDARY = 3
    POST_SECONDARY = 4
    SHORT_TERTIARY = 5
    BACHELOR = 6
    MASTER = 7
    DOCTORAL = 8
