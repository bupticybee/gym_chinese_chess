from gym.envs.registration import register

register(
    id='cchess-v0',
    entry_point='gym_chinese_chess.envs:ChineseChessEnv',
)
