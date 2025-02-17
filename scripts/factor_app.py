from database.factor_calculator import FullFactorCalculator
from database.scheduler import SchedulerManager


if __name__ == "__main__":
    calculator = FullFactorCalculator()
    
    import argparse
    parser = argparse.ArgumentParser(description='多因子数据系统')
    parser.add_argument('--init', action='store_true', help='初始化全量历史数据')
    parser.add_argument('--daemon', action='store_true', help='启动定时服务')
    
    args = parser.parse_args()
    
    if args.init:
        print("开始初始化全量历史数据...")
        calculator.initialize_hist_data()
        print("历史数据初始化完成！")
        
    if args.daemon:
        print("启动定时更新服务...")
        scheduler = SchedulerManager(calculator)
        scheduler.setup_schedules()
        scheduler.start()
