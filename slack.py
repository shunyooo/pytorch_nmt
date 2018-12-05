from slacker import Slacker
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

API_KEY = config['slack']['API_KEY']
CHANNEL = config['slack']['CHANNEL']
slacker = Slacker(API_KEY)


def post(_str, pretext=None, title=None, color="good", channel=None):
    if channel is None:
        channel = CHANNEL
    slacker.chat.post_message(channel,
                              as_user=True,
                              attachments=[
                                  {
                                      "title": title,
                                      "pretext": pretext,
                                      "color": color,
                                      "text": _str
                                  }
                              ]
                              )


def show_help():
    info = """
postでメッセージを送れます。
_strは枠内のテキスト
pretextは枠外のテキスト
titleは枠内太字のテキスト
colorは枠の色。danger, good, warningなど
channelはデフォルト syunyooo-bot-notify
channelでは、botは招待しておく必要がある。
"""
    return info
