# tasks/ball_objectnav_task.py

from habitat.tasks.nav.object_nav_task import ObjectNavTask

class BallObjectNavTask(ObjectNavTask):
    def __init__(self, config):
        super().__init__(config)
        tm = self._sim.get_object_template_manager()
        self._ball_tid = tm.load_configs("data/objects/ball")[0]
        print("🔴 BallObjectNavTask ready with template id", self._ball_tid)

    def reset(self, episode):
        # 1) reset normal : position agent, navmesh, etc.
        obs = super().reset(episode)

        # 2) spawn la balle, protégé contre toute exception
        try:
            goal_pos = episode.goals[0].position
            self._ball_oid = self._sim.add_object(
                self._ball_tid,
                position=goal_pos,
                rotation=[0, 0, 0, 1]
            )
        except Exception as e:
            # affiche la stack pour comprendre, mais ne crashe plus
            import traceback
            print("⚠️  Erreur en spawn ball :", e)
            traceback.print_exc()

        return obs
