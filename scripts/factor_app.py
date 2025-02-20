from database.factor_calculator import FullFactorCalculator
from database.scheduler import SchedulerManager
import pandas as pd

if __name__ == "__main__":
    calculator = FullFactorCalculator()
    
    import argparse
    parser = argparse.ArgumentParser(description='多因子数据系统')
    parser.add_argument('--init', action='store_true', help='初始化全量历史数据')
    parser.add_argument('--daemon', action='store_true', help='启动定时服务')
    parser.add_argument('--opt', action='store_true', help='策略优化获取优化参数')
    
    args = parser.parse_args()

    if args.opt:
        print("开始优化策略...")
    
    if args.init:
        print("开始初始化全量历史数据...")
        calculator.initialize_hist_data()
        print("历史数据初始化完成！")
        
    if args.daemon:
        print("启动定时更新服务...")
        scheduler = SchedulerManager(calculator)
        start_date = pd.to_datetime("today").date()
        scheduler.setup_schedules(start_date)
        scheduler.start()
