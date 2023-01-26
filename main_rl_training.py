from Robots.charlier import Charlier
from Robots.RLBotMinimal import ReinforcementLearningBotMinimal
from AI.RLBotTraining_minimal import Training

botList = []
#your bot
bot = ReinforcementLearningBotMinimal
botList.append((bot, True))
#enemy
bot = Charlier
botList.append((bot, False))
x = 500
y = 700
rlbot_training = Training(x, y, botList, no_graphics = True)