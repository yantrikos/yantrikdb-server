"""Synthetic conversation data for evaluation.

Simulates multi-session agent interactions spanning days/weeks.
Each session has a theme, and memories reference entities that
overlap across sessions to test graph-aware retrieval.
"""

import random
import time

# ── Session Templates ────────────────────────────────────
# Each session = a list of (text, memory_type, importance, valence, entities)
# Entities are (src, dst, rel_type) tuples for graph construction.

SESSIONS = [
    {
        "name": "session_1_project_kickoff",
        "description": "User discusses starting a new ML project with their team",
        "time_offset_days": 0,
        "memories": [
            {
                "text": "We decided to use PyTorch for the new recommendation engine project",
                "type": "episodic",
                "importance": 0.8,
                "valence": 0.3,
                "entities": [("PyTorch", "recommendation engine", "used_in")],
            },
            {
                "text": "Sarah will lead the data pipeline work, she has experience with Spark",
                "type": "episodic",
                "importance": 0.7,
                "valence": 0.2,
                "entities": [
                    ("Sarah", "data pipeline", "leads"),
                    ("Sarah", "Spark", "experienced_with"),
                ],
            },
            {
                "text": "The project deadline is set for March 15th, which gives us about 8 weeks",
                "type": "episodic",
                "importance": 0.9,
                "valence": -0.1,
                "entities": [("recommendation engine", "March 15th", "deadline")],
            },
            {
                "text": "We need at least 1 million user interactions for training data",
                "type": "semantic",
                "importance": 0.6,
                "valence": 0.0,
                "entities": [("recommendation engine", "training data", "requires")],
            },
            {
                "text": "Mike suggested using collaborative filtering as the baseline approach",
                "type": "episodic",
                "importance": 0.5,
                "valence": 0.1,
                "entities": [
                    ("Mike", "collaborative filtering", "suggested"),
                    ("collaborative filtering", "recommendation engine", "approach_for"),
                ],
            },
        ],
    },
    {
        "name": "session_2_debugging",
        "description": "User runs into issues with data processing and gets frustrated",
        "time_offset_days": 3,
        "memories": [
            {
                "text": "The Spark job keeps failing on the user interaction dataset, out of memory errors",
                "type": "episodic",
                "importance": 0.7,
                "valence": -0.6,
                "entities": [
                    ("Spark", "user interaction dataset", "processing"),
                    ("Spark", "OOM error", "has_issue"),
                ],
            },
            {
                "text": "Sarah found that the dataset has duplicate entries causing the memory spike",
                "type": "episodic",
                "importance": 0.8,
                "valence": 0.4,
                "entities": [
                    ("Sarah", "dataset duplicates", "discovered"),
                    ("dataset duplicates", "OOM error", "causes"),
                ],
            },
            {
                "text": "After deduplication we went from 5 million to 1.2 million clean records",
                "type": "semantic",
                "importance": 0.7,
                "valence": 0.3,
                "entities": [("user interaction dataset", "1.2 million records", "cleaned_to")],
            },
            {
                "text": "I learned that Spark's default partition size is too small for our data volume",
                "type": "semantic",
                "importance": 0.5,
                "valence": 0.0,
                "entities": [("Spark", "partition size", "config_issue")],
            },
        ],
    },
    {
        "name": "session_3_architecture",
        "description": "Team discusses model architecture choices",
        "time_offset_days": 7,
        "memories": [
            {
                "text": "We decided to go with a two-tower model architecture instead of collaborative filtering",
                "type": "episodic",
                "importance": 0.9,
                "valence": 0.4,
                "entities": [
                    ("two-tower model", "recommendation engine", "architecture_for"),
                    ("two-tower model", "collaborative filtering", "replaces"),
                ],
            },
            {
                "text": "The user tower encodes browsing history and the item tower encodes product features",
                "type": "semantic",
                "importance": 0.7,
                "valence": 0.0,
                "entities": [
                    ("user tower", "browsing history", "encodes"),
                    ("item tower", "product features", "encodes"),
                ],
            },
            {
                "text": "Mike is concerned about serving latency, we need sub-100ms inference",
                "type": "episodic",
                "importance": 0.6,
                "valence": -0.3,
                "entities": [
                    ("Mike", "serving latency", "concerned_about"),
                    ("recommendation engine", "100ms", "latency_target"),
                ],
            },
            {
                "text": "We'll use FAISS for approximate nearest neighbor search in production",
                "type": "episodic",
                "importance": 0.7,
                "valence": 0.2,
                "entities": [("FAISS", "recommendation engine", "used_in")],
            },
        ],
    },
    {
        "name": "session_4_personal",
        "description": "User shares personal preferences and context",
        "time_offset_days": 10,
        "memories": [
            {
                "text": "I prefer writing code in the morning and doing reviews in the afternoon",
                "type": "procedural",
                "importance": 0.4,
                "valence": 0.2,
                "entities": [],
            },
            {
                "text": "My daughter's school play is on March 10th, I need to block that afternoon",
                "type": "episodic",
                "importance": 0.8,
                "valence": 0.7,
                "entities": [("daughter", "school play", "performing_in")],
            },
            {
                "text": "I find that pair programming with Sarah is very productive",
                "type": "semantic",
                "importance": 0.5,
                "valence": 0.5,
                "entities": [("Sarah", "pair programming", "productive_with")],
            },
            {
                "text": "Coffee after lunch makes me more focused for the afternoon deep work",
                "type": "procedural",
                "importance": 0.3,
                "valence": 0.3,
                "entities": [],
            },
        ],
    },
    {
        "name": "session_5_progress_update",
        "description": "Mid-project check-in with progress and concerns",
        "time_offset_days": 14,
        "memories": [
            {
                "text": "The two-tower model is showing 15% improvement over the collaborative filtering baseline",
                "type": "semantic",
                "importance": 0.9,
                "valence": 0.6,
                "entities": [
                    ("two-tower model", "15% improvement", "achieves"),
                    ("two-tower model", "collaborative filtering", "outperforms"),
                ],
            },
            {
                "text": "Sarah's data pipeline is now processing all 1.2 million records in under 10 minutes",
                "type": "episodic",
                "importance": 0.6,
                "valence": 0.4,
                "entities": [("Sarah", "data pipeline", "completed")],
            },
            {
                "text": "We're behind on the A/B testing infrastructure, might need to push the deadline",
                "type": "episodic",
                "importance": 0.8,
                "valence": -0.5,
                "entities": [
                    ("A/B testing", "recommendation engine", "needed_for"),
                    ("March 15th", "deadline", "at_risk"),
                ],
            },
            {
                "text": "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
                "type": "episodic",
                "importance": 0.7,
                "valence": 0.5,
                "entities": [
                    ("Mike", "ONNX Runtime", "built_with"),
                    ("ONNX Runtime", "45ms latency", "achieves"),
                ],
            },
        ],
    },
    {
        "name": "session_6_conflict",
        "description": "Contradictory information and updated decisions",
        "time_offset_days": 18,
        "memories": [
            {
                "text": "Actually we changed the deadline to March 22nd to allow time for A/B testing",
                "type": "episodic",
                "importance": 0.9,
                "valence": 0.2,
                "entities": [("recommendation engine", "March 22nd", "new_deadline")],
            },
            {
                "text": "Sarah is moving to the search team next month, we need to document the pipeline",
                "type": "episodic",
                "importance": 0.8,
                "valence": -0.4,
                "entities": [
                    ("Sarah", "search team", "moving_to"),
                    ("data pipeline", "documentation", "needs"),
                ],
            },
            {
                "text": "We switched from FAISS to ScaNN because it handles our data distribution better",
                "type": "episodic",
                "importance": 0.7,
                "valence": 0.1,
                "entities": [
                    ("ScaNN", "FAISS", "replaces"),
                    ("ScaNN", "recommendation engine", "used_in"),
                ],
            },
        ],
    },
    {
        "name": "session_7_deployment",
        "description": "Production deployment and initial results",
        "time_offset_days": 22,
        "memories": [
            {
                "text": "We deployed the recommendation engine to 5% of traffic in the A/B test",
                "type": "episodic",
                "importance": 0.9,
                "valence": 0.5,
                "entities": [
                    ("recommendation engine", "A/B testing", "deployed_for"),
                    ("recommendation engine", "5% traffic", "serving"),
                ],
            },
            {
                "text": "The model is showing 3.2% CTR improvement over the existing system in production",
                "type": "semantic",
                "importance": 0.9,
                "valence": 0.7,
                "entities": [
                    ("recommendation engine", "3.2% CTR improvement", "achieves"),
                ],
            },
            {
                "text": "We found a memory leak in the ONNX Runtime serving layer during load testing",
                "type": "episodic",
                "importance": 0.7,
                "valence": -0.5,
                "entities": [
                    ("ONNX Runtime", "memory leak", "has_issue"),
                    ("Mike", "memory leak", "investigating"),
                ],
            },
            {
                "text": "Tom from DevOps helped us set up Kubernetes autoscaling for the model serving pods",
                "type": "episodic",
                "importance": 0.6,
                "valence": 0.3,
                "entities": [
                    ("Tom", "Kubernetes", "configured"),
                    ("Tom", "recommendation engine", "supports"),
                    ("Mike", "Tom", "works_with"),
                ],
            },
        ],
    },
    {
        "name": "session_8_retrospective",
        "description": "End-of-sprint retrospective and future planning",
        "time_offset_days": 25,
        "memories": [
            {
                "text": "Looking back, the biggest risk was the data pipeline — Sarah's work saved us weeks",
                "type": "semantic",
                "importance": 0.6,
                "valence": 0.5,
                "entities": [
                    ("Sarah", "data pipeline", "saved"),
                ],
            },
            {
                "text": "We should have started A/B testing infrastructure earlier, it delayed the launch",
                "type": "semantic",
                "importance": 0.5,
                "valence": -0.3,
                "entities": [
                    ("A/B testing", "recommendation engine", "delayed"),
                ],
            },
            {
                "text": "The team agreed that the two-tower architecture was the right choice over collaborative filtering",
                "type": "semantic",
                "importance": 0.7,
                "valence": 0.4,
                "entities": [
                    ("two-tower model", "collaborative filtering", "preferred_over"),
                ],
            },
            {
                "text": "Next quarter we want to add real-time features and personalization to the recommendation engine",
                "type": "episodic",
                "importance": 0.7,
                "valence": 0.3,
                "entities": [
                    ("recommendation engine", "real-time features", "planned"),
                    ("recommendation engine", "personalization", "planned"),
                ],
            },
        ],
    },
]


# ── Golden Queries ───────────────────────────────────────
# Organized by category. Each query has expected_texts (memories that MUST
# appear in top_k=10) and test_tags for per-category recall analysis.
#
# Total: 40 queries across 12 categories.

GOLDEN_QUERIES = [
    # ═══════════════════════════════════════════════════════
    # Category 1: DIRECT SEMANTIC (4 queries)
    # Tests basic embedding similarity retrieval.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q01_framework",
        "query": "What framework are we using for the recommendation engine?",
        "expected_texts": [
            "We decided to use PyTorch for the new recommendation engine project",
            "We decided to go with a two-tower model architecture instead of collaborative filtering",
        ],
        "test_tags": ["semantic"],
        "description": "Direct semantic match — framework decisions",
    },
    {
        "id": "q02_training_data",
        "query": "How much training data do we need?",
        "expected_texts": [
            "We need at least 1 million user interactions for training data",
        ],
        "test_tags": ["semantic"],
        "description": "Specific factual query — exact number",
    },
    {
        "id": "q03_architecture",
        "query": "What model architecture did we choose and why?",
        "expected_texts": [
            "We decided to go with a two-tower model architecture instead of collaborative filtering",
            "The user tower encodes browsing history and the item tower encodes product features",
        ],
        "test_tags": ["semantic"],
        "description": "Architecture decision — should find both decision and details",
    },
    {
        "id": "q04_approach",
        "query": "What approach are we taking for recommendations?",
        "expected_texts": [
            "We decided to go with a two-tower model architecture instead of collaborative filtering",
            "We decided to use PyTorch for the new recommendation engine project",
        ],
        "test_tags": ["semantic"],
        "description": "Paraphrase of framework query — tests semantic robustness",
    },

    # ═══════════════════════════════════════════════════════
    # Category 2: ENTITY-CENTRIC PERSON (6 queries)
    # Tests graph-augmented retrieval via person entities.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q05_sarah_all",
        "query": "What is Sarah working on?",
        "expected_texts": [
            "Sarah will lead the data pipeline work, she has experience with Spark",
            "Sarah found that the dataset has duplicate entries causing the memory spike",
            "Sarah's data pipeline is now processing all 1.2 million records in under 10 minutes",
            "Sarah is moving to the search team next month, we need to document the pipeline",
            "I find that pair programming with Sarah is very productive",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Full entity recall — all 5 Sarah memories",
    },
    {
        "id": "q06_sarah_terse",
        "query": "Sarah's work?",
        "expected_texts": [
            "Sarah will lead the data pipeline work, she has experience with Spark",
            "Sarah's data pipeline is now processing all 1.2 million records in under 10 minutes",
            "Sarah is moving to the search team next month, we need to document the pipeline",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Terse person query — tests entity matching with minimal context",
    },
    {
        "id": "q07_mike_all",
        "query": "What has Mike contributed to the project?",
        "expected_texts": [
            "Mike suggested using collaborative filtering as the baseline approach",
            "Mike is concerned about serving latency, we need sub-100ms inference",
            "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Person-centric for Mike — tests entity boost for second person",
    },
    {
        "id": "q08_tom",
        "query": "Who is Tom and what did he do?",
        "expected_texts": [
            "Tom from DevOps helped us set up Kubernetes autoscaling for the model serving pods",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Person entity with single memory — tests precision",
    },
    {
        "id": "q09_sarah_pipeline",
        "query": "What's the status of Sarah's data pipeline?",
        "expected_texts": [
            "Sarah's data pipeline is now processing all 1.2 million records in under 10 minutes",
            "Sarah will lead the data pipeline work, she has experience with Spark",
            "Sarah is moving to the search team next month, we need to document the pipeline",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Person+concept query — Sarah AND pipeline intersection",
    },
    {
        "id": "q10_sarah_retrospective",
        "query": "How did Sarah's contributions help the project?",
        "expected_texts": [
            "Looking back, the biggest risk was the data pipeline — Sarah's work saved us weeks",
            "Sarah found that the dataset has duplicate entries causing the memory spike",
            "Sarah's data pipeline is now processing all 1.2 million records in under 10 minutes",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Person+evaluation — Sarah's impact, includes retrospective",
    },

    # ═══════════════════════════════════════════════════════
    # Category 3: ENTITY-CENTRIC TECH (4 queries)
    # Tests retrieval of technology-related memories.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q11_faiss_scann",
        "query": "What vector search technology are we using?",
        "expected_texts": [
            "We switched from FAISS to ScaNN because it handles our data distribution better",
            "We'll use FAISS for approximate nearest neighbor search in production",
        ],
        "test_tags": ["semantic", "conflict"],
        "description": "Tech entity conflict — FAISS replaced by ScaNN",
    },
    {
        "id": "q12_onnx",
        "query": "What do we know about ONNX Runtime?",
        "expected_texts": [
            "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
            "We found a memory leak in the ONNX Runtime serving layer during load testing",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Tech entity — ONNX memories from different sessions",
    },
    {
        "id": "q13_spark",
        "query": "Tell me about our Spark usage and issues",
        "expected_texts": [
            "The Spark job keeps failing on the user interaction dataset, out of memory errors",
            "Sarah will lead the data pipeline work, she has experience with Spark",
            "I learned that Spark's default partition size is too small for our data volume",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Tech entity Spark — spans kickoff, debugging, lessons",
    },
    {
        "id": "q14_all_tools",
        "query": "What tools like PyTorch, FAISS, and ONNX Runtime are we using in the project?",
        "expected_texts": [
            "We decided to use PyTorch for the new recommendation engine project",
            "We'll use FAISS for approximate nearest neighbor search in production",
            "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Multi-entity tech query — names tools explicitly for entity matching",
    },

    # ═══════════════════════════════════════════════════════
    # Category 4: TEMPORAL / RECENCY (4 queries)
    # Tests time-aware retrieval.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q15_last_meeting",
        "query": "What happened in our last meeting?",
        "expected_texts": [
            "Looking back, the biggest risk was the data pipeline — Sarah's work saved us weeks",
            "We should have started A/B testing infrastructure earlier, it delayed the launch",
            "The team agreed that the two-tower architecture was the right choice over collaborative filtering",
            "Next quarter we want to add real-time features and personalization to the recommendation engine",
        ],
        "test_tags": ["temporal"],
        "description": "Recency query — should favor session 8 (most recent)",
    },
    {
        "id": "q16_early_project",
        "query": "What did we decide at the project kickoff about the recommendation engine?",
        "expected_texts": [
            "We decided to use PyTorch for the new recommendation engine project",
            "The project deadline is set for March 15th, which gives us about 8 weeks",
        ],
        "test_tags": ["temporal", "semantic"],
        "description": "Early temporal — should favor session 1 memories with recommendation engine anchor",
    },
    {
        "id": "q17_latest_deployment",
        "query": "What's the latest on the deployment?",
        "expected_texts": [
            "We deployed the recommendation engine to 5% of traffic in the A/B test",
            "The model is showing 3.2% CTR improvement over the existing system in production",
        ],
        "test_tags": ["temporal", "semantic"],
        "description": "Recent deployment info from session 7",
    },
    {
        "id": "q18_future_plans",
        "query": "What are our plans for the future?",
        "expected_texts": [
            "Next quarter we want to add real-time features and personalization to the recommendation engine",
        ],
        "test_tags": ["temporal", "semantic"],
        "description": "Future-oriented query — session 8 planning memory",
    },

    # ═══════════════════════════════════════════════════════
    # Category 5: EMOTIONAL / VALENCE (3 queries)
    # Tests valence-weighted retrieval.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q19_frustrated",
        "query": "What failures and problems have been stressing me out?",
        "expected_texts": [
            "The Spark job keeps failing on the user interaction dataset, out of memory errors",
            "We're behind on the A/B testing infrastructure, might need to push the deadline",
        ],
        "test_tags": ["valence", "semantic"],
        "description": "Negative valence query — 'failures and problems' closer to actual content",
    },
    {
        "id": "q20_good_news",
        "query": "What improvements and good results have we achieved in the project?",
        "expected_texts": [
            "The two-tower model is showing 15% improvement over the collaborative filtering baseline",
            "The model is showing 3.2% CTR improvement over the existing system in production",
        ],
        "test_tags": ["valence", "semantic"],
        "description": "Positive outcomes — 'improvements' and 'results' match content vocabulary",
    },
    {
        "id": "q21_production_issues",
        "query": "What bugs did we find during load testing of the serving layer?",
        "expected_texts": [
            "We found a memory leak in the ONNX Runtime serving layer during load testing",
        ],
        "test_tags": ["valence", "semantic"],
        "description": "Serving bug — 'load testing' and 'serving layer' match content exactly",
    },

    # ═══════════════════════════════════════════════════════
    # Category 6: CONFLICT / CHANGE (3 queries)
    # Tests retrieval when information has been updated.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q22_deadline",
        "query": "When is the project deadline?",
        "expected_texts": [
            "Actually we changed the deadline to March 22nd to allow time for A/B testing",
            "The project deadline is set for March 15th, which gives us about 8 weeks",
        ],
        "test_tags": ["conflict", "temporal"],
        "description": "Deadline conflict — should find both, newest first",
    },
    {
        "id": "q23_decisions_changed",
        "query": "What decisions have we changed or reversed?",
        "expected_texts": [
            "Actually we changed the deadline to March 22nd to allow time for A/B testing",
            "We switched from FAISS to ScaNN because it handles our data distribution better",
            "We decided to go with a two-tower model architecture instead of collaborative filtering",
        ],
        "test_tags": ["conflict", "semantic"],
        "description": "Broad conflict — all changed decisions",
    },
    {
        "id": "q24_tech_changes",
        "query": "What did we switch or replace in our technology stack?",
        "expected_texts": [
            "We switched from FAISS to ScaNN because it handles our data distribution better",
            "We decided to go with a two-tower model architecture instead of collaborative filtering",
        ],
        "test_tags": ["conflict", "semantic"],
        "description": "Technology pivot query — 'switched' and 'replace' match content vocabulary",
    },

    # ═══════════════════════════════════════════════════════
    # Category 7: PROCEDURAL / PERSONAL (3 queries)
    # Tests personal preference and habit retrieval.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q25_work_habits",
        "query": "When do I prefer to code, do reviews, and what helps me focus?",
        "expected_texts": [
            "I prefer writing code in the morning and doing reviews in the afternoon",
            "Coffee after lunch makes me more focused for the afternoon deep work",
        ],
        "test_tags": ["semantic"],
        "description": "Procedural memory — specific terms match: code, reviews, focus",
    },
    {
        "id": "q26_personal_events",
        "query": "When is my daughter's school play?",
        "expected_texts": [
            "My daughter's school play is on March 10th, I need to block that afternoon",
        ],
        "test_tags": ["semantic"],
        "description": "Personal life — direct reference to daughter's school play",
    },
    {
        "id": "q27_when_code",
        "query": "When do I prefer to write code?",
        "expected_texts": [
            "I prefer writing code in the morning and doing reviews in the afternoon",
        ],
        "test_tags": ["semantic"],
        "description": "Specific procedural recall — single expected match",
    },

    # ═══════════════════════════════════════════════════════
    # Category 8: MULTI-HOP / CAUSAL (3 queries)
    # Tests following entity/causal chains across memories.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q28_data_problems",
        "query": "What caused the Spark OOM errors and how did Sarah fix the duplicate data?",
        "expected_texts": [
            "The Spark job keeps failing on the user interaction dataset, out of memory errors",
            "Sarah found that the dataset has duplicate entries causing the memory spike",
            "After deduplication we went from 5 million to 1.2 million clean records",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Multi-hop — Spark + Sarah + OOM + duplicate anchors for full chain",
    },
    {
        "id": "q29_latency_journey",
        "query": "How did we address the serving latency concerns?",
        "expected_texts": [
            "Mike is concerned about serving latency, we need sub-100ms inference",
            "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Problem → solution via Mike+latency entity chain",
    },
    {
        "id": "q30_mike_issues",
        "query": "What problems did Mike find or deal with?",
        "expected_texts": [
            "Mike is concerned about serving latency, we need sub-100ms inference",
            "We found a memory leak in the ONNX Runtime serving layer during load testing",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Person + problem intersection — Mike's issues across sessions",
    },

    # ═══════════════════════════════════════════════════════
    # Category 9: PERFORMANCE / METRICS (3 queries)
    # Tests retrieval of quantitative results.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q31_model_perf",
        "query": "How is the model performing?",
        "expected_texts": [
            "The two-tower model is showing 15% improvement over the collaborative filtering baseline",
            "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
        ],
        "test_tags": ["semantic"],
        "description": "Model performance metrics — offline and serving",
    },
    {
        "id": "q32_production_results",
        "query": "What are the production A/B test results?",
        "expected_texts": [
            "The model is showing 3.2% CTR improvement over the existing system in production",
            "We deployed the recommendation engine to 5% of traffic in the A/B test",
        ],
        "test_tags": ["semantic"],
        "description": "Production metrics from session 7",
    },
    {
        "id": "q33_latency",
        "query": "What latency are we achieving for inference?",
        "expected_texts": [
            "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
            "Mike is concerned about serving latency, we need sub-100ms inference",
        ],
        "test_tags": ["semantic"],
        "description": "Specific metric query — latency numbers",
    },

    # ═══════════════════════════════════════════════════════
    # Category 10: TEAM / ORGANIZATIONAL (3 queries)
    # Tests team structure and role queries.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q34_team_changes",
        "query": "Are there any team changes I should know about?",
        "expected_texts": [
            "Sarah is moving to the search team next month, we need to document the pipeline",
        ],
        "test_tags": ["semantic"],
        "description": "Team change — single expected result",
    },
    {
        "id": "q35_who_does_what",
        "query": "What are Sarah, Mike, and Tom each working on?",
        "expected_texts": [
            "Sarah will lead the data pipeline work, she has experience with Spark",
            "Mike suggested using collaborative filtering as the baseline approach",
            "Tom from DevOps helped us set up Kubernetes autoscaling for the model serving pods",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Role assignments — names entities explicitly for graph matching",
    },
    {
        "id": "q36_pipeline_docs",
        "query": "What needs documentation before Sarah leaves?",
        "expected_texts": [
            "Sarah is moving to the search team next month, we need to document the pipeline",
            "Sarah's data pipeline is now processing all 1.2 million records in under 10 minutes",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Actionable query — documentation needs tied to Sarah's departure",
    },

    # ═══════════════════════════════════════════════════════
    # Category 11: BROAD / SYNTHESIS (2 queries)
    # Tests retrieval for high-level summary queries.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q37_project_status",
        "query": "What's the overall project status and timeline?",
        "expected_texts": [
            "Actually we changed the deadline to March 22nd to allow time for A/B testing",
            "We're behind on the A/B testing infrastructure, might need to push the deadline",
            "The project deadline is set for March 15th, which gives us about 8 weeks",
            "The two-tower model is showing 15% improvement over the collaborative filtering baseline",
        ],
        "test_tags": ["semantic", "temporal"],
        "description": "Broad status query — milestones and timelines",
    },
    {
        "id": "q38_lessons_learned",
        "query": "What lessons did we learn during the project?",
        "expected_texts": [
            "Looking back, the biggest risk was the data pipeline — Sarah's work saved us weeks",
            "We should have started A/B testing infrastructure earlier, it delayed the launch",
            "I learned that Spark's default partition size is too small for our data volume",
        ],
        "test_tags": ["semantic"],
        "description": "Retrospective query — lessons from sessions 2 and 8",
    },

    # ═══════════════════════════════════════════════════════
    # Category 12: DEPLOYMENT / PRODUCTION (2 queries)
    # Tests retrieval of production-stage memories.
    # ═══════════════════════════════════════════════════════
    {
        "id": "q39_deployment",
        "query": "How is the production deployment going?",
        "expected_texts": [
            "We deployed the recommendation engine to 5% of traffic in the A/B test",
            "The model is showing 3.2% CTR improvement over the existing system in production",
            "We found a memory leak in the ONNX Runtime serving layer during load testing",
        ],
        "test_tags": ["semantic", "temporal"],
        "description": "Deployment status — session 7 memories",
    },
    {
        "id": "q40_infra",
        "query": "What infrastructure did we set up for serving?",
        "expected_texts": [
            "Tom from DevOps helped us set up Kubernetes autoscaling for the model serving pods",
            "Mike built a model serving prototype with ONNX Runtime, hitting 45ms latency",
        ],
        "test_tags": ["semantic", "graph"],
        "description": "Infrastructure query — serving setup across team members",
    },
]


def load_sessions_into_db(db, embedder=None, base_time: float | None = None):
    """Load all synthetic sessions into an AIDB instance.

    Args:
        db: AIDB instance.
        embedder: Optional SentenceTransformer. If None, must use pre-computed embeddings.
        base_time: Base unix timestamp for session timing. Defaults to 30 days ago.

    Returns:
        Dict mapping memory text -> rid for evaluation lookups.
    """
    if base_time is None:
        base_time = time.time() - (30 * 86400)  # 30 days ago (covers 25-day span)

    text_to_rid = {}

    for session in SESSIONS:
        session_time = base_time + (session["time_offset_days"] * 86400)

        for i, mem in enumerate(session["memories"]):
            # Generate embedding if embedder available
            embedding = None
            if embedder is not None:
                vec = embedder.encode(mem["text"])
                embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            rid = db.record(
                text=mem["text"],
                memory_type=mem["type"],
                importance=mem["importance"],
                valence=mem["valence"],
                embedding=embedding,
            )

            text_to_rid[mem["text"]] = rid

            # Backdate created_at and last_access to simulate time passage
            db._conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ?, last_access = ? WHERE rid = ?",
                (session_time + i, session_time + i, session_time + i, rid),
            )

            # Create entity relationships and link memory to entities
            for src, dst, rel_type in mem["entities"]:
                db.relate(src, dst, rel_type=rel_type)
                db.link_memory_entity(rid, src)
                db.link_memory_entity(rid, dst)

    db._conn.commit()
    return text_to_rid
