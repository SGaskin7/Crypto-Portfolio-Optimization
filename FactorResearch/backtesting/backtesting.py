from Backtester import Backtester
from FactorModelCVaR_DecisionMaker import FactorModelCVaRDecisionMaker



FactorModelCVaRDecisionMaker = DecisionMaker(asset_list)
FactorModelCVaRBacktester = Backtester(FactorModelCVaRDecisionMaker)

FactorModelCVaRBacktester.LoadData()
print(FactorModelCVaRBacktester.data.head)


# SVMMVOBacktester = Backtester()

