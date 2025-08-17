#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored Intelligent Memory Management System Core Interface Demo
Demonstrates core APIs: instantiation, write_memory_auto (sync/async), read_memory_auto, get_status
"""

import os
import time
import datetime
from src.core.memory_manager import MemManagerAgent
from src.utils.config import ConfigLoader
from src.utils.logger import get_logger, setup_logging

# Logging initialization
setup_logging()
logger = get_logger(__name__)


def async_callback(result):
    """Asynchronous Write Completion Callback Function"""
    print(f"   🔔 Callback Notification: Write Complete - Success: {result.get('success', False)}")
    if result.get('success'):
        prelim_result = result.get('preliminary_result', {})
        action = prelim_result.get('action', 'unknown')
        print(f"      📝 Write Action: {action}")
        if action == 'updated':
            similarity = prelim_result.get('similarity_score', 0)
            print(f"      📊 Similarity: {similarity:.3f}")


def main():
    """Main demonstration function"""
    print("🚀 Refactored Intelligent Memory Management System Core API Demo")
    print("=" * 60)
    print("Includes: instantiation, write_memory_auto (sync/async), read_memory_auto, get_status")
    print("=" * 60)

    try:
        # 1. System initialization
        print("\n1. System initialization")
        print("-" * 40)

        storage_path = "demo_memory"
        config_file = "config.txt"

        if not os.path.exists(config_file):
            print(f"Configuration file not found: {config_file}")
            return

        print(f"Storage path: {storage_path}")
        print(f"Configuration file: {config_file}")

        # Create Asynchronous Memory Manager
        agent = MemManagerAgent(
            storage_path=storage_path, 
            config_file=config_file,
            enable_async=True,  # Enable Asynchronous Mode
            worker_threads=2
        )
        print(f"✅ System initialization completed")
        print(f"   Similarity threshold: {agent.similarity_threshold}")
        print(f"   Max tokens: {agent.max_tokens}")
        print(f"   Async mode: {agent.enable_async}")
        print(f"   Worker threads: {agent.worker_threads}")

        # 2. Demonstrate Asynchronous Write Function
        print("\n2. Demonstrate Asynchronous Write Function")
        print("-" * 40)

        async_memories = [
            # Technical Learning Memory
            {
                "text": "Today I learned the basics of quantum computing. Quantum bits (qubits) are the basic units of quantum computing. Unlike classical bits",
                "priority": 1
            },
            {
                "text": "I deeply studied Python's asynchronous programming",
                "priority": 2
            },
            {
                "text": "I learned deep learning techniques in machine learning",
                "priority": 1
            },
            {
                "text": "I studied the basic principles of blockchain technology",
                "priority": 0
            },
            {
                "text": "I learned Docker containerization technology and understood the difference between containers and virtual machines. Docker achieves standardized deployment of applications through images and containers",
                "priority": 1
            },
            
            # Life Skills Memory
            {
                "text": "I attended a cooking course and learned basic French cuisine techniques. French cooking emphasizes the freshness of ingredients and precision in cooking. I learned to make basic French stock",
                "priority": 0
            },
            {
                "text": "I learned basic photography composition techniques",
                "priority": 0
            },
            {
                "text": "I attended a yoga class and learned basic poses and breathing techniques. Yoga not only improves body flexibility but also helps relax the mind and improve sleep quality.",
                "priority": 0
            },
            {
                "text": "I learned time management techniques",
                "priority": 1
            },
            {
                "text": "I attended public speaking training and learned how to overcome nervousness and improve expression skills. I mastered techniques for body language",
                "priority": 1
            },
            
            # Reading and Learning Memory
            {
                "text": "I read "One Hundred Years of Solitude",
                "priority": 2
            },
            {
                "text": "阅读了《人类简史》，作者尤瓦尔·赫拉利从认知革命、农业革命到科技革命，重新解读了人类历史的发展脉络。",
                "priority": 1
            },
            {
                "text": "I read "Thinking",
                "priority": 1
            },
            {
                "text": "I read the book "Principles." Ray Dalio shared his life and work principles",
                "priority": 0
            },
            {
                "text": "I read the science fiction novel "The Three-Body Problem." Liu Cixin",
                "priority": 1
            },
            
            # Work Project Memory
            {
                "text": "I completed the requirements analysis for the company's new product and had in-depth discussions with product managers and designers. Core functional modules were determined",
                "priority": 2
            },
            {
                "text": "I attended a technical team meeting and discussed system architecture optimization plans. It was decided to adopt a microservices architecture to improve system scalability and maintainability.",
                "priority": 1
            },
            {
                "text": "I conducted a project progress report with the client",
                "priority": 1
            },
            {
                "text": "I completed code review work",
                "priority": 0
            },
            {
                "text": "I attended an industry technical conference",
                "priority": 1
            },
            
            # Social Activity Memory
            {
                "text": "I gathered with old friends and shared our respective work and life updates. Communication between friends can bring new ideas and inspiration",
                "priority": 0
            },
            {
                "text": "I participated in community volunteer activities",
                "priority": 0
            },
            {
                "text": "I participated in team building activities with colleagues",
                "priority": 0
            },
            {
                "text": "I attended a book club and discussed the themes and significance of the book "To Live" with fellow readers. The collision of different viewpoints made the reading experience more enriching.",
                "priority": 0
            },
            {
                "text": "I spent a pleasant weekend with family",
                "priority": 1
            },
            
            # Healthy Living Memory
            {
                "text": "I started to persist in running for 30 minutes every day. Running not only exercises the body but also releases stress and improves mental state.",
                "priority": 1
            },
            {
                "text": "I adjusted my sleep schedule to ensure 7-8 hours of sleep daily. Adequate sleep is important for physical health and work efficiency.",
                "priority": 1
            },
            {
                "text": "I learned about nutritional matching and started paying attention to dietary balance. Reasonable nutritional intake is the foundation for maintaining health.",
                "priority": 0
            },
            {
                "text": "I attended a mental health lecture and learned how to manage stress and emotions",
                "priority": 1
            },
            {
                "text": "I started practicing meditation",
                "priority": 0
            }
        ]

        print(f"\n📝 Asynchronous Write {len(async_memories)} Memories")
        print("-" * 40)

        request_ids = []
        for i, memory in enumerate(async_memories, 1):
            print(f"\nAsynchronous Write Memory {i}: {memory['text'][:30]}...")
            print(f"  Priority: {memory['priority']}")

            try:
                result = agent.write_memory_auto(
                    text=memory['text'],
                    update_memoir_all=True,  # Auto-generate Memoir
                    callback=async_callback,
                    priority=memory['priority']
                )

                if result.get('success', False):
                    print(f"✅ Asynchronous Write Request Submitted")
                    print(f"   Request ID: {result['request_id']}")
                    print(f"   Status: {result['status']}")
                    print(f"   Queue Position: {result['queue_position']}")
                    print(f"   Estimated Wait Time: {result['estimated_wait_time']}Seconds")
                    print(f"   Text Preview: {result['text_preview']}")
                    
                    request_ids.append(result['request_id'])
                else:
                    print(f"❌ Asynchronous Write Failed: {result.get('error', 'unknown error')}")

            except Exception as e:
                print(f"❌ Asynchronous Write Exception: {e}")

        # 3. Demonstrate Request Status Query
        print("\n3. Demonstrate Request Status Query")
        print("-" * 40)

        for i, request_id in enumerate(request_ids, 1):
            print(f"\n🔍 Query Request {i} Status: {request_id}")
            
            # Wait for a While to Let Request Start Processing
            time.sleep(0.5)
            
            try:
                status = agent.get_request_status(request_id)
                if status.get('success', False):
                    print(f"   Status: {status['status']}")
                    print(f"   Priority: {status['priority']}")
                    print(f"   Submission Time: {datetime.datetime.fromtimestamp(status['timestamp']).strftime('%H:%M:%S')}")
                    
                    if 'start_time' in status:
                        print(f"   Start Time: {datetime.datetime.fromtimestamp(status['start_time']).strftime('%H:%M:%S')}")
                    
                    if 'processing_time' in status:
                        print(f"   Processing Time: {status['processing_time']:.2f}Seconds")
                    
                    if 'error' in status:
                        print(f"   Error Message: {status['error']}")
                else:
                    print(f"   ❌ Status查询失败: {status.get('error', 'unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Status查询异常: {e}")

        # 4. Wait for All Asynchronous Requests to Complete
        print("\n4. Wait for All Asynchronous Requests to Complete")
        print("-" * 40)
        
        print("⏳ Wait for Requests in Queue to Complete Processing...")
        agent.wait_for_completion()
        print("✅ All Asynchronous Requests Have Been Processed")

        # 5. 查看最终Status
        print("\n5. 查看最终处理Status")
        print("-" * 40)

        for i, request_id in enumerate(request_ids, 1):
            print(f"\n📊 Request {i} 最终Status: {request_id}")
            
            try:
                final_status = agent.get_request_status(request_id)
                if final_status.get('success', False):
                    print(f"   最终Status: {final_status['status']}")
                    print(f"   Processing Time: {final_status.get('processing_time', 0):.2f}Seconds")
                    
                    if 'result' in final_status:
                        result_data = final_status['result']
                        if result_data.get('success'):
                            prelim_result = result_data.get('preliminary_result', {})
                            action = prelim_result.get('action', 'unknown')
                            print(f"   Write Action: {action}")
                            if action == 'updated':
                                similarity = prelim_result.get('similarity_score', 0)
                                print(f"   Similarity: {similarity:.3f}")
                        else:
                            print(f"   Processing Failed: {result_data.get('error', 'unknown error')}")
                else:
                    print(f"   ❌ Status查询失败: {final_status.get('error', 'unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Status查询异常: {e}")

        # 6. Demonstrate Synchronous Write (Comparison)
        print("\n6. Demonstrate Synchronous Write (Comparison)")
        print("-" * 40)

        sync_memories = [
            "Today I learned Python's object-oriented programming",
            "I learned Python's decorator pattern",
            "I studied Python's asynchronous programming",
            "I learned the basics of data structures and algorithms",
            "I studied software design patterns",
            "I learned to use the version control system Git",
            "I studied database design and optimization",
            "I learned the basics of web development",
            "I studied the basic concepts of network security",
            "I learned the basic concepts of cloud computing"
        ]

        print(f"\n📝 Synchronous Write {len(sync_memories)} Memories")
        print("-" * 40)

        # Create New Synchronous Mode Manager
        sync_agent = MemManagerAgent(
            storage_path="demo_memory",  # Keep Consistent with Asynchronous Write
            config_file=config_file,
            enable_async=False
        )

        for i, text in enumerate(sync_memories, 1):
            print(f"\nSynchronous Write Memory {i}: {text[:30]}...")
            
            start_time = time.time()
            try:
                result = sync_agent.write_memory_auto(
                    text=text,
                    update_memoir_all=True  # Auto-generate Memoir
                )
                end_time = time.time()
                
                if result.get('success', False):
                    prelim_result = result.get('preliminary_result', {})
                    action = prelim_result.get('action', 'unknown')
                    mem_id = prelim_result.get('mem_id', 'unknown')
                    print(f"✅ 同步Write Complete")
                    print(f"   Action: {action}")
                    print(f"   Memory ID: {mem_id}")
                    print(f"   Time Taken: {end_time - start_time:.2f}Seconds")

                    if action == 'updated':
                        similarity = prelim_result.get('similarity_score', 0)
                        print(f"   Similarity: {similarity:.3f}")
                else:
                    print(f"❌ Synchronous Write Failed: {result.get('error', 'unknown error')}")
                    if 'preliminary_result' in result:
                        prelim_result = result['preliminary_result']
                        action = prelim_result.get('action', 'unknown')
                        print(f"   Action: {action}")

            except Exception as e:
                print(f"❌ Synchronous Write Exception: {e}")
        
        sync_agent.shutdown()

        # 7. 演示异步Status管理
        print("\n7. 演示异步Status管理")
        print("-" * 40)

        try:
            # 获取所有请求Status
            all_status = agent.get_all_request_status()
            print(f"📋 Total Request Count: {all_status['total_requests']}")
            
            # 统计不同Status的请求
            status_counts = {}
            for request_id, status_info in all_status['requests'].items():
                status = status_info.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print("📊 Status统计:")
            for status, count in status_counts.items():
                print(f"   {status}: {count} Items")
            
            # Clean Up Completed Requests
            print("\n🧹 Clean Up Completed RequestsStatus...")
            agent.cleanup_completed_requests(max_age_hours=1)
            
            # 再次获取Status
            all_status_after = agent.get_all_request_status()
            print(f"📋 Total Request Count After Cleanup: {all_status_after['total_requests']}")
            
        except Exception as e:
            print(f"❌ Status管理异常: {e}")

        # 8. Test intelligent search (read_memory_auto)
        print("\n8. Test intelligent search (read_memory_auto)")
        print("-" * 40)

        # Test different types of search queries
        search_queries = [
            "Python programming",  # Should find related memory
            "Quantum computing",    # Should find unrelated memory
            "Decorator",      # Should find related memory
            "French cuisine",    # Should find unrelated memory
            "Asynchronous programming",    # Should find related memory
            "What Did I Do This Year",
            "What Did I Do Today",
        ]

        for i, query in enumerate(search_queries, 1):
            print(f"\nSearch {i}: '{query}'")

            try:
                results = agent.read_memory_auto(query, top_k=3)

                if results['success']:
                    print(f"✅ Search type: {results['search_type']}")
                    print(f"   Found {len(results['results'])} related memories")

                    for j, result in enumerate(results['results'], 1):
                        mem_cell = result['mem_cell']
                        similarity = result.get('similarity_score', 0)
                        print(f"   {j}. Similarity: {similarity:.3f}")
                        print(f"      Summary: {mem_cell.summary}")
                        print(
                            f"      Created: {datetime.datetime.fromtimestamp(mem_cell.create_time).strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"❌ Search failed: {results.get('error', 'unknown error')}")

            except Exception as e:
                print(f"❌ Search exception: {e}")

        # 9. Test time query
        print("\n9. Test time query")
        print("-" * 40)

        # Get current time information
        current_time = datetime.datetime.now()
        current_year = current_time.year
        current_month = current_time.month
        current_day = current_time.day

        time_queries = [
            f"{current_year}Year",
            f"{current_year}Year{current_month}Month",
            f"{current_year}Year{current_month}Month{current_day}Day",
            "Today",
            "This Month"
        ]

        for i, time_query in enumerate(time_queries, 1):
            print(f"\nTime query {i}: '{time_query}'")

            try:
                results = agent.read_memory_auto(time_query, top_k=5)

                if results['success']:
                    print(f"✅ Search type: {results['search_type']}")
                    print(f"   Found {len(results['results'])} related memories")

                    for j, result in enumerate(results['results'], 1):
                        mem_cell = result['mem_cell']
                        if 'similarity_score' in result:
                            similarity = result['similarity_score']
                            print(f"   {j}. Similarity: {similarity:.3f}")
                        else:
                            print(f"   {j}. Time match")
                        print(f"      Summary: {mem_cell.summary}")
                        print(
                            f"      Created: {datetime.datetime.fromtimestamp(mem_cell.create_time).strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"❌ Time query failed: {results.get('error', 'unknown error')}")

            except Exception as e:
                print(f"❌ Time query exception: {e}")

        # 10. Test get_status_summary function
        print("\n10. Test get_status_summary function")
        print("-" * 40)

        try:
            summary = agent.get_status_summary()
            if summary['success']:
                print(f"✅ Status summary retrieved successfully")
                
                # Display Basic Statistics
                print(f"   Storage path: {summary.get('storage_path', 'unknown')}")
                print(f"   Similarity threshold: {summary.get('similarity_threshold', 'unknown')}")
                print(f"   Max tokens: {summary.get('max_tokens', 'unknown')}")

                # Display Module Statistics
                if 'preliminary_memory' in summary:
                    prelim = summary['preliminary_memory']
                    print(f"   Preliminary memory: {prelim.get('memory_count', 0)} entries")
                    print(f"     Storage size: {prelim.get('total_size_mb', 0)} MB")

                if 'memoir' in summary:
                    memoir = summary['memoir']
                    print(f"   Memoir memory: {memoir.get('total_memoirs', 0)} entries")
                    print(f"     Storage size: {memoir.get('total_size_mb', 0)} MB")

                # Display Asynchronous Processing Statistics
                if 'async_summary' in summary:
                    async_summary = summary['async_summary']
                    print(f"   Async processing:")
                    print(f"     Enabled: {async_summary.get('async_enabled', False)}")
                    print(f"     Queue size: {async_summary.get('queue_size', 0)}")
                    print(f"     Total requests: {async_summary.get('total_requests', 0)}")
                    print(f"     Processed requests: {async_summary.get('processed_requests', 0)}")
                    print(f"     Failed requests: {async_summary.get('failed_requests', 0)}")
                    print(f"     Success rate: {async_summary.get('success_rate', 0):.1f}%")
                    print(f"     Average processing time: {async_summary.get('average_processing_time', 0):.2f}Seconds")
            else:
                print(f"❌ Status summary retrieval failed: {summary.get('error', 'unknown error')}")

        except Exception as e:
            print(f"❌ Status summary retrieval exception: {e}")

        # 11. Shutdown System
        print("\n11. Shutdown System")
        print("-" * 40)
        
        print("🔒 Shutting Down Asynchronous Memory Manager...")
        agent.shutdown(wait=True)
        print("✅ System Has Been Safely Shut Down")

        print("\n" + "=" * 60)
        print("🎉 Intelligent Memory Management System Demo Completed!")
        print("=" * 60)
        print("Tested the following features:")
        print("✅ System initialization (with async support)")
        print("✅ Async write_memory_auto with callbacks")
        print("✅ Request status tracking and querying")
        print("✅ Sync vs async comparison")
        print("✅ Async status management and cleanup")
        print("✅ Intelligent search test (read_memory_auto)")
        print("✅ Time query test")
        print("✅ System status summary with async stats")
        print("✅ Graceful shutdown")

        print(f"\nStorage directory: {agent.storage_path}")
        print("You can view this directory to understand the system's storage structure")

    except Exception as e:
        print(f"\n❌ Error occurred during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
