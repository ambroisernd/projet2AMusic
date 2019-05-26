def parse_duration(s):
    """return string duration to float"""
    if "/" in s:
        return float(s.split("/")[0]) / float(s.split("/")[-1])
    else:
        return float(s)


def RepresentsInt(s):
    """helper fct to check if string is int"""
    try:
        int(s)
        return True
    except ValueError:
        return False
