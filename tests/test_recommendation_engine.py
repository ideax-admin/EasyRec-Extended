import unittest

class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        # Set up the recommendation engine instance (This is pseudocode; replace with actual initialization)
        self.engine = RecommendationEngine()

    def test_recommend_item(self):
        # Test case to recommend an item based on user preferences
        user_id = 1
        item_id = 101
        recommendation = self.engine.recommend_item(user_id)
        self.assertIn(item_id, recommendation,
                      "Recommended items should include item 101 for user 1")

    def test_no_active_users(self):
        # Test behavior when there are no active users
        self.engine.clear_users()  # Assuming there's a method to clear users
        self.assertRaises(NoActiveUsersError, self.engine.recommend_item, 1)

    def test_item_similarity(self):
        # Test the similarity function of items
        item_a = 101
        item_b = 102
        similarity = self.engine.compute_similarity(item_a, item_b)
        self.assertGreaterEqual(similarity, 0,
                                "Similarity should be a non-negative value")

    def test_recommendation_order(self):
        # Test that recommendations are ordered by relevance
        user_id = 1
        recommendations = self.engine.recommend_item(user_id)
        self.assertEqual(recommendations, sorted(recommendations, key=lambda x: x.relevance),
                         "Recommendations should be sorted by relevance")

    def tearDown(self):
        # Clean up after each test case
        self.engine.reset()


if __name__ == '__main__':
    unittest.main()