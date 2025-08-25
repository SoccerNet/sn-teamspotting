import zipfile
import json

def LoadJsonFromZip(zippedFile, JsonPath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        # print(filename)
        JsonPath = JsonPath.replace("\\", "/")
        with z.open(JsonPath) as f:
            data = f.read()
            d = json.loads(data.decode("utf-8"))

    return d

def getListGames(split = "test", dataset = "SoccerNet-Ball"):
    assert split in ["train", "val", "test", "challenge"], "split should be either 'train', 'val', 'test', or 'challenge'"
    assert dataset in ["SoccerNet-Ball"], "only implemented for SoccerNet-Ball"

    SNB_games = {
        'train': ["england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
                "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
                "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
                "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest"],
        'val' : ["england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End"],
        'test': ["england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
                "england_efl/2019-2020/2019-10-01 - Reading - Fulham"],
        'challenge': ["england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
                "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City"]
        }
    
    return SNB_games[split]

EVENT_DICTIONARY_BALL = {"PASS":0, "DRIVE":1, "HEADER":2, "HIGH PASS":3, "OUT":4, "CROSS":5, "THROW IN":6, "SHOT":7, "BALL PLAYER BLOCK":8, 
                        "PLAYER SUCCESSFUL TACKLE":9, "FREE KICK":10, "GOAL":11}