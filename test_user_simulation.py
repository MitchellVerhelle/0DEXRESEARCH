import unittest
import numpy as np
from users import RegularUser, SybilUser
from user_pool import UserPool

class TestUserSimulation(unittest.TestCase):

    def test_regular_user_preTGE(self):
        user = RegularUser(wealth=1000, user_id=1, user_size='small')
        initial_points = user.airdrop_points
        user.step('PreTGE')
        self.assertGreater(user.airdrop_points, initial_points,
                           "RegularUser should accumulate airdrop points during PreTGE.")

    def test_regular_user_TGE(self):
        user = RegularUser(wealth=1000, user_id=2, user_size='medium')
        user.airdrop_points = 10
        user.step('TGE')
        self.assertEqual(user.tokens, 10,
                         "Default AirdropPolicy should assign tokens equal to airdrop points.")

    def test_regular_user_postTGE(self):
        user = RegularUser(wealth=1000, user_id=3, user_size='large')
        user.step('PostTGE')
        self.assertIn(user.active, [True, False],
                      "User active status should be a boolean after PostTGE.")

    def test_sybil_user_behavior(self):
        user = SybilUser(wealth=500, user_id=4)
        user.step('PreTGE')
        user.step('TGE')
        user.step('PostTGE')
        self.assertFalse(user.active,
                         "SybilUser should become inactive after TGE.")
        self.assertGreaterEqual(user.tokens, 0,
                                "SybilUser should receive tokens at TGE even with minimal activity.")

    def test_user_pool_generation(self):
        pool = UserPool(num_users=1000)
        self.assertEqual(len(pool.users), 1000, "UserPool should generate the correct number of users.")
        num_sybil = sum(isinstance(u, SybilUser) for u in pool.users)
        num_regular = sum(isinstance(u, RegularUser) for u in pool.users)
        self.assertEqual(num_sybil + num_regular, 1000,
                         "UserPool should contain only RegularUser or SybilUser instances.")
        self.assertTrue(250 < num_sybil < 350,
                        "Sybil percentage should be around 30% of total users.")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

# in root, run
# python -m unittest test_user_simulation.py
# ... or run
# pytest test_user_simulation.py