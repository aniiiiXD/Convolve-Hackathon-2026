import unittest
from unittest.mock import MagicMock, ANY
import logging
import sys
import os

# Mock Qdrant Client before importing ranking_agent
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()

from medisync.model_agents.ranking_agent import ReRankerModel

class TestRankingAgent(unittest.TestCase):
    def setUp(self):
        self.reranker = ReRankerModel()
        self.reranker.client = MagicMock()
        
    def test_rerank_calls_query_points_correctly(self):
        # Setup mocks
        mock_query_points = self.reranker.client.query_points
        mock_result = MagicMock()
        mock_result.points = ["point1", "point2"]
        mock_query_points.return_value = mock_result
        
        # Test Input
        dense_vec = [0.1] * 768
        sparse_vec = MagicMock()
        sparse_vec.indices = [1, 2]
        sparse_vec.values = [0.5, 0.6]
        
        # Action
        res = self.reranker.rerank_with_qdrant(
            collection_name="test_coll",
            query="test query",
            query_vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=5
        )
        
        # Assertions
        # 1. Check query_points was called
        mock_query_points.assert_called_once()
        
        # 2. Check arguments
        call_args = mock_query_points.call_args[1]
        self.assertEqual(call_args['collection_name'], "test_coll")
        self.assertEqual(call_args['limit'], 5)
        self.assertTrue(call_args['with_payload'])
        
        # 3. Check Prefetch (Hybrid)
        prefetch = call_args['prefetch']
        self.assertEqual(len(prefetch), 2) # Dense + Sparse
        
        # Verify call structure matches implicit RRF expectations
        # (Mocking objects makes deep inspection harder but we verified the list length)

        print("Test Passed: query_points called with Hybrid Prefetch")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
