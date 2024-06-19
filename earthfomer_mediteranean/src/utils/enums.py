from enum import Enum

class Variables(Enum):
    PRECIPITATION = "tp"
    TEMPERATURE = "t2m"
    OLR = "ttr"

class StackType(Enum):
    DAILY = 1
    WEEKLY = 7
    BIWEEKLY = 14
    MONTHLY = 30
    BIMONTHY = 60

class Resolution(Enum):
    WEEKLY = "week"
    DAILY = "day"
    MONTHLY = "month"
    SEASON = "season"
    YEARLY = "year"