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
    """异步写入完成回调函数"""
    print(f"   🔔 回调通知: 写入完成 - 成功: {result.get('success', False)}")
    if result.get('success'):
        prelim_result = result.get('preliminary_result', {})
        action = prelim_result.get('action', 'unknown')
        print(f"      📝 写入动作: {action}")
        if action == 'updated':
            similarity = prelim_result.get('similarity_score', 0)
            print(f"      📊 相似度: {similarity:.3f}")


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

        # 创建异步内存管理器
        agent = MemManagerAgent(
            storage_path=storage_path, 
            config_file=config_file,
            enable_async=True,  # 启用异步模式
            worker_threads=2
        )
        print(f"✅ System initialization completed")
        print(f"   Similarity threshold: {agent.similarity_threshold}")
        print(f"   Max tokens: {agent.max_tokens}")
        print(f"   Async mode: {agent.enable_async}")
        print(f"   Worker threads: {agent.worker_threads}")

        # 2. 演示异步写入功能
        print("\n2. 演示异步写入功能")
        print("-" * 40)

        async_memories = [
            # 技术学习类记忆
            {
                "text": "今天学习了量子计算的基础知识。量子比特（qubit）是量子计算的基本单位，与经典比特不同，它可以同时处于多个状态的叠加。量子纠缠是量子计算的核心特性，两个或多个量子比特可以形成纠缠态，即使相距很远也能瞬间影响彼此的状态。",
                "priority": 1
            },
            {
                "text": "深入研究了Python的异步编程，使用asyncio库来处理并发任务。异步编程可以显著提高I/O密集型应用的性能，通过协程实现非阻塞操作。学会了使用async/await语法，以及如何管理异步上下文。",
                "priority": 2
            },
            {
                "text": "学习了机器学习中的深度学习技术，特别是卷积神经网络（CNN）在图像识别中的应用。理解了卷积层、池化层和全连接层的作用，以及反向传播算法的工作原理。",
                "priority": 1
            },
            {
                "text": "研究了区块链技术的基本原理，包括去中心化、共识机制、密码学哈希等核心概念。比特币作为第一个区块链应用，展示了分布式账本技术的潜力。",
                "priority": 0
            },
            {
                "text": "学习了Docker容器化技术，理解了容器与虚拟机的区别。Docker通过镜像和容器实现了应用程序的标准化部署，大大简化了开发环境的配置和部署流程。",
                "priority": 1
            },
            
            # 生活技能类记忆
            {
                "text": "参加了烹饪课程，学习了法式料理的基本技巧。法式烹饪强调食材的新鲜度和烹饪的精确性。我学会了制作基础的法式高汤（stock），这是很多法式菜肴的基础。",
                "priority": 0
            },
            {
                "text": "学习了摄影的基本构图技巧，包括三分法、对称构图、引导线等。理解了光圈、快门速度和ISO的关系，以及如何在不同光线条件下调整参数。",
                "priority": 0
            },
            {
                "text": "参加了瑜伽课程，学习了基础的体式和呼吸技巧。瑜伽不仅能提高身体的柔韧性，还能帮助放松心情，改善睡眠质量。",
                "priority": 0
            },
            {
                "text": "学习了时间管理技巧，包括番茄工作法、四象限法则等。合理的时间管理能显著提高工作效率，减少压力和焦虑。",
                "priority": 1
            },
            {
                "text": "参加了公共演讲培训，学习了如何克服紧张情绪，提高表达能力。掌握了肢体语言、语速控制和观众互动的技巧。",
                "priority": 1
            },
            
            # 阅读学习类记忆
            {
                "text": "阅读了《百年孤独》这本魔幻现实主义文学经典。作者加西亚·马尔克斯通过布恩迪亚家族七代人的故事，展现了拉丁美洲的历史变迁。",
                "priority": 2
            },
            {
                "text": "阅读了《人类简史》，作者尤瓦尔·赫拉利从认知革命、农业革命到科技革命，重新解读了人类历史的发展脉络。",
                "priority": 1
            },
            {
                "text": "阅读了《思考，快与慢》，丹尼尔·卡尼曼详细介绍了人类思维的两种模式：快速直觉和慢速理性，以及认知偏差对决策的影响。",
                "priority": 1
            },
            {
                "text": "阅读了《原则》这本书，雷·达里奥分享了他的人生和工作原则，强调了透明度和独立思考的重要性。",
                "priority": 0
            },
            {
                "text": "阅读了《三体》科幻小说，刘慈欣通过三体文明与地球文明的接触，探讨了宇宙文明、科技发展和人性等深刻主题。",
                "priority": 1
            },
            
            # 工作项目类记忆
            {
                "text": "完成了公司新产品的需求分析，与产品经理和设计师进行了深入讨论。确定了核心功能模块，制定了开发计划和时间节点。",
                "priority": 2
            },
            {
                "text": "参加了技术团队会议，讨论了系统架构的优化方案。决定采用微服务架构来提升系统的可扩展性和维护性。",
                "priority": 1
            },
            {
                "text": "与客户进行了项目进度汇报，展示了已完成的功能模块和下一步的开发计划。客户对项目进展表示满意。",
                "priority": 1
            },
            {
                "text": "完成了代码审查工作，检查了团队成员的代码质量，提出了改进建议。代码审查是保证软件质量的重要环节。",
                "priority": 0
            },
            {
                "text": "参加了行业技术会议，听取了关于人工智能发展趋势的演讲，了解了最新的技术动态和应用案例。",
                "priority": 1
            },
            
            # 社交活动类记忆
            {
                "text": "与老朋友聚会，分享了各自的工作和生活近况。朋友间的交流能带来新的想法和启发，是人生中重要的精神支持。",
                "priority": 0
            },
            {
                "text": "参加了社区志愿者活动，帮助老年人学习使用智能手机。通过志愿服务，感受到了帮助他人的快乐和成就感。",
                "priority": 0
            },
            {
                "text": "与同事一起参加了团建活动，通过团队游戏增进了彼此的了解，提升了团队凝聚力。",
                "priority": 0
            },
            {
                "text": "参加了读书会，与书友们讨论了《活着》这本书的主题和意义。不同观点的碰撞让阅读体验更加丰富。",
                "priority": 0
            },
            {
                "text": "与家人一起度过了愉快的周末时光，一起做饭、看电影，享受了温馨的家庭时光。",
                "priority": 1
            },
            
            # 健康生活类记忆
            {
                "text": "开始坚持每天跑步30分钟，跑步不仅能锻炼身体，还能释放压力，提高精神状态。",
                "priority": 1
            },
            {
                "text": "调整了作息时间，保证每天7-8小时的睡眠。充足的睡眠对身体健康和工作效率都很重要。",
                "priority": 1
            },
            {
                "text": "学习了营养搭配知识，开始注意饮食的均衡性。合理的营养摄入是保持健康的基础。",
                "priority": 0
            },
            {
                "text": "参加了心理健康讲座，学习了如何管理压力和情绪，保持积极的心态。",
                "priority": 1
            },
            {
                "text": "开始练习冥想，每天花10分钟进行正念练习，这有助于提高专注力和情绪管理能力。",
                "priority": 0
            }
        ]

        print(f"\n📝 异步写入 {len(async_memories)} 个记忆")
        print("-" * 40)

        request_ids = []
        for i, memory in enumerate(async_memories, 1):
            print(f"\n异步写入记忆 {i}: {memory['text'][:30]}...")
            print(f"  优先级: {memory['priority']}")

            try:
                result = agent.write_memory_auto(
                    text=memory['text'],
                    update_memoir_all=True,  # 自动生成 memoir
                    callback=async_callback,
                    priority=memory['priority']
                )

                if result.get('success', False):
                    print(f"✅ 异步写入请求已提交")
                    print(f"   请求ID: {result['request_id']}")
                    print(f"   状态: {result['status']}")
                    print(f"   队列位置: {result['queue_position']}")
                    print(f"   估算等待时间: {result['estimated_wait_time']}秒")
                    print(f"   文本预览: {result['text_preview']}")
                    
                    request_ids.append(result['request_id'])
                else:
                    print(f"❌ 异步写入失败: {result.get('error', 'unknown error')}")

            except Exception as e:
                print(f"❌ 异步写入异常: {e}")

        # 3. 演示请求状态查询
        print("\n3. 演示请求状态查询")
        print("-" * 40)

        for i, request_id in enumerate(request_ids, 1):
            print(f"\n🔍 查询请求 {i} 状态: {request_id}")
            
            # 等待一段时间让请求开始处理
            time.sleep(0.5)
            
            try:
                status = agent.get_request_status(request_id)
                if status.get('success', False):
                    print(f"   状态: {status['status']}")
                    print(f"   优先级: {status['priority']}")
                    print(f"   提交时间: {datetime.datetime.fromtimestamp(status['timestamp']).strftime('%H:%M:%S')}")
                    
                    if 'start_time' in status:
                        print(f"   开始时间: {datetime.datetime.fromtimestamp(status['start_time']).strftime('%H:%M:%S')}")
                    
                    if 'processing_time' in status:
                        print(f"   处理时间: {status['processing_time']:.2f}秒")
                    
                    if 'error' in status:
                        print(f"   错误信息: {status['error']}")
                else:
                    print(f"   ❌ 状态查询失败: {status.get('error', 'unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ 状态查询异常: {e}")

        # 4. 等待所有异步请求完成
        print("\n4. 等待所有异步请求完成")
        print("-" * 40)
        
        print("⏳ 等待队列中的请求处理完成...")
        agent.wait_for_completion()
        print("✅ 所有异步请求已处理完成")

        # 5. 查看最终状态
        print("\n5. 查看最终处理状态")
        print("-" * 40)

        for i, request_id in enumerate(request_ids, 1):
            print(f"\n📊 请求 {i} 最终状态: {request_id}")
            
            try:
                final_status = agent.get_request_status(request_id)
                if final_status.get('success', False):
                    print(f"   最终状态: {final_status['status']}")
                    print(f"   处理时间: {final_status.get('processing_time', 0):.2f}秒")
                    
                    if 'result' in final_status:
                        result_data = final_status['result']
                        if result_data.get('success'):
                            prelim_result = result_data.get('preliminary_result', {})
                            action = prelim_result.get('action', 'unknown')
                            print(f"   写入动作: {action}")
                            if action == 'updated':
                                similarity = prelim_result.get('similarity_score', 0)
                                print(f"   相似度: {similarity:.3f}")
                        else:
                            print(f"   处理失败: {result_data.get('error', 'unknown error')}")
                else:
                    print(f"   ❌ 状态查询失败: {final_status.get('error', 'unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ 状态查询异常: {e}")

        # 6. 演示同步写入（对比）
        print("\n6. 演示同步写入（对比）")
        print("-" * 40)

        sync_memories = [
            "今天学习了Python的面向对象编程，理解了类、继承、多态和封装的概念。面向对象编程能更好地组织代码结构，提高代码的可维护性和复用性。",
            "学习了Python的装饰器模式，这是一个非常强大的特性，可以用于日志记录、性能监控、权限验证等功能。装饰器让代码更加简洁和优雅。",
            "研究了Python的异步编程，使用asyncio库来处理并发任务，提高了程序的性能。异步编程特别适合I/O密集型应用，如网络请求和文件操作。",
            "学习了数据结构和算法的基础知识，包括数组、链表、栈、队列、树等。良好的算法设计能显著提升程序的执行效率。",
            "研究了软件设计模式，包括单例模式、工厂模式、观察者模式等。设计模式是解决常见软件设计问题的标准方案。",
            "学习了版本控制系统Git的使用，包括分支管理、合并策略、冲突解决等。Git是现代软件开发中不可或缺的工具。",
            "研究了数据库设计和优化，学习了关系型数据库的范式理论，以及如何设计高效的数据库结构。",
            "学习了Web开发的基础知识，包括HTML、CSS、JavaScript等前端技术，以及HTTP协议和RESTful API设计。",
            "研究了网络安全的基本概念，包括加密算法、身份认证、访问控制等。网络安全在当今数字化时代越来越重要。",
            "学习了云计算的基本概念，包括IaaS、PaaS、SaaS等服务模式，以及云原生应用的设计原则。"
        ]

        print(f"\n📝 同步写入 {len(sync_memories)} 个记忆")
        print("-" * 40)

        # 新建同步模式管理器
        sync_agent = MemManagerAgent(
            storage_path="demo_memory",  # 与异步写入保持一致
            config_file=config_file,
            enable_async=False
        )

        for i, text in enumerate(sync_memories, 1):
            print(f"\n同步写入记忆 {i}: {text[:30]}...")
            
            start_time = time.time()
            try:
                result = sync_agent.write_memory_auto(
                    text=text,
                    update_memoir_all=True  # 自动生成 memoir
                )
                end_time = time.time()
                
                if result.get('success', False):
                    prelim_result = result.get('preliminary_result', {})
                    action = prelim_result.get('action', 'unknown')
                    mem_id = prelim_result.get('mem_id', 'unknown')
                    print(f"✅ 同步写入完成")
                    print(f"   动作: {action}")
                    print(f"   内存ID: {mem_id}")
                    print(f"   耗时: {end_time - start_time:.2f}秒")

                    if action == 'updated':
                        similarity = prelim_result.get('similarity_score', 0)
                        print(f"   相似度: {similarity:.3f}")
                else:
                    print(f"❌ 同步写入失败: {result.get('error', 'unknown error')}")
                    if 'preliminary_result' in result:
                        prelim_result = result['preliminary_result']
                        action = prelim_result.get('action', 'unknown')
                        print(f"   动作: {action}")

            except Exception as e:
                print(f"❌ 同步写入异常: {e}")
        
        sync_agent.shutdown()

        # 7. 演示异步状态管理
        print("\n7. 演示异步状态管理")
        print("-" * 40)

        try:
            # 获取所有请求状态
            all_status = agent.get_all_request_status()
            print(f"📋 总请求数: {all_status['total_requests']}")
            
            # 统计不同状态的请求
            status_counts = {}
            for request_id, status_info in all_status['requests'].items():
                status = status_info.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print("📊 状态统计:")
            for status, count in status_counts.items():
                print(f"   {status}: {count} 个")
            
            # 清理已完成的请求
            print("\n🧹 清理已完成的请求状态...")
            agent.cleanup_completed_requests(max_age_hours=1)
            
            # 再次获取状态
            all_status_after = agent.get_all_request_status()
            print(f"📋 清理后总请求数: {all_status_after['total_requests']}")
            
        except Exception as e:
            print(f"❌ 状态管理异常: {e}")

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
            "今年干了啥",
            "今天干了啥",
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
            f"{current_year}年",
            f"{current_year}年{current_month}月",
            f"{current_year}年{current_month}月{current_day}日",
            "今天",
            "这个月"
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
                
                # 显示基本统计信息
                print(f"   Storage path: {summary.get('storage_path', 'unknown')}")
                print(f"   Similarity threshold: {summary.get('similarity_threshold', 'unknown')}")
                print(f"   Max tokens: {summary.get('max_tokens', 'unknown')}")

                # 显示模块统计信息
                if 'preliminary_memory' in summary:
                    prelim = summary['preliminary_memory']
                    print(f"   Preliminary memory: {prelim.get('memory_count', 0)} entries")
                    print(f"     Storage size: {prelim.get('total_size_mb', 0)} MB")

                if 'memoir' in summary:
                    memoir = summary['memoir']
                    print(f"   Memoir memory: {memoir.get('total_memoirs', 0)} entries")
                    print(f"     Storage size: {memoir.get('total_size_mb', 0)} MB")

                # 显示异步处理统计信息
                if 'async_summary' in summary:
                    async_summary = summary['async_summary']
                    print(f"   Async processing:")
                    print(f"     Enabled: {async_summary.get('async_enabled', False)}")
                    print(f"     Queue size: {async_summary.get('queue_size', 0)}")
                    print(f"     Total requests: {async_summary.get('total_requests', 0)}")
                    print(f"     Processed requests: {async_summary.get('processed_requests', 0)}")
                    print(f"     Failed requests: {async_summary.get('failed_requests', 0)}")
                    print(f"     Success rate: {async_summary.get('success_rate', 0):.1f}%")
                    print(f"     Average processing time: {async_summary.get('average_processing_time', 0):.2f}秒")
            else:
                print(f"❌ Status summary retrieval failed: {summary.get('error', 'unknown error')}")

        except Exception as e:
            print(f"❌ Status summary retrieval exception: {e}")

        # 11. 关闭系统
        print("\n11. 关闭系统")
        print("-" * 40)
        
        print("🔒 正在关闭异步内存管理器...")
        agent.shutdown(wait=True)
        print("✅ 系统已安全关闭")

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
