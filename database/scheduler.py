from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

class SchedulerManager:
    def __init__(self, calculator):
        self.calculator = calculator
        self.scheduler = BlockingScheduler()
        
    def setup_schedules(self):
        # 每日收盘后任务
        self.scheduler.add_job(
            self.calculator.daily_update,
            trigger=CronTrigger(hour=18, minute=30),
            name='daily_update'
        )
        
        # 每周更新宏观数据
        self.scheduler.add_job(
            self.calculator._update_macro_data,
            trigger=CronTrigger(day_of_week='mon', hour=6),
            name='weekly_macro'
        )
        
        # 每月初更新财务报表
        self.scheduler.add_job(
            self.calculator._check_financial_reports,
            trigger=CronTrigger(day=1, hour=7),
            name='monthly_financial'
        )
        
        # 每季度更新质量因子
        self.scheduler.add_job(
            self.calculator.batch_update_quality_factors,
            trigger=CronTrigger(month='*/3', day=1, hour=8),
            name='quarterly_quality'
        )

    def start(self):
        self.scheduler.start()
