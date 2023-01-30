from Robots.charlier import Charlier
from Robots.RLBot_Tjark_Nitsche import ReinforcementLearningBotTjark

from AI.RLBotTraining_minimal import Training

botList = []
#your bot
bot = ReinforcementLearningBotTjark

botList.append((bot, True))
#enemy
bot = Charlier
botList.append((bot, False))
x = 500
y = 700
rlbot_training = Training(x, y, botList, no_graphics = True)
