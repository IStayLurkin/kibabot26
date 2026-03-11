from tasks.health_tasks import HealthTasks


class TaskManager:
    def __init__(self, bot):
        self.bot = bot
        self.health_tasks = HealthTasks(bot)

    def start_all(self):
        self.health_tasks.start_all()

    def stop_all(self):
        self.health_tasks.stop_all()
