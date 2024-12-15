import datetime


def s2ts(s: int) -> str:
    t_s = str(datetime.timedelta(seconds=round(s)))
    ts = t_s.split(":")
    return ts[0] + ":" + ts[1] + ":" + ts[2]
