from stable_baselines3.common.callbacks import BaseCallback

class SuccessRateCallback(BaseCallback):
    """
    Ne regarde QUE les infos contenant 'is_success' pour
    compter les épisodes et les succès.
    """
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.successes = 0
        self.episodes  = 0

    def _on_step(self) -> bool:
        # Pour chaque env, on cherche les infos de fin d'épisode
        for info in self.locals["infos"]:
            if "is_success" in info:
                self.episodes  += 1
                self.successes += int(info["is_success"])
        return True

    def _on_rollout_end(self) -> None:
        if self.episodes > 0:
            rate = self.successes / self.episodes
            # Clé 'rollout/' pour l’affichage console SB3
            self.logger.record("rollout/ep_success_rate", rate)
            # reset
            self.successes = 0
            self.episodes  = 0
