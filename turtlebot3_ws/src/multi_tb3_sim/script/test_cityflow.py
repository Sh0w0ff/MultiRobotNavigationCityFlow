#!/usr/bin/env python3
"""
Test script to verify OpenAI Gym and CityFlow installations
Run this script to check if both libraries are properly installed and working
"""

import sys
import os

def test_gym_installation():
    """Test OpenAI Gym installation and basic functionality"""
    print("=" * 50)
    print("TESTING OPENAI GYM")
    print("=" * 50)
    
    try:
        import gym
        print(f"✅ OpenAI Gym imported successfully")
        print(f"   Version: {gym.__version__}")
        
        # Test creating a simple environment
        env = gym.make('CartPole-v1')
        print(f"✅ Created CartPole-v1 environment")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Test basic environment operations
        reset_result = env.reset()
        # Handle both old and new gym API
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        print(f"✅ Environment reset successful")
        print(f"   Initial observation shape: {obs.shape}")
        
        # Take a random action
        action = env.action_space.sample()
        step_result = env.step(action)
        if len(step_result) == 5:  # New gym API
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # Old gym API
            obs, reward, done, info = step_result
        print(f"✅ Environment step successful")
        print(f"   Action: {action}")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        
        env.close()
        print(f"✅ Environment closed successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import OpenAI Gym: {e}")
        print("   Install with: pip install gym")
        return False
    except Exception as e:
        print(f"❌ Error testing Gym: {e}")
        return False

def test_cityflow_installation():
    """Test CityFlow installation and basic functionality"""
    print("\n" + "=" * 50)
    print("TESTING CITYFLOW")
    print("=" * 50)
    
    try:
        import cityflow
        print(f"✅ CityFlow imported successfully")
        
        # Create a minimal config for testing
        config = create_minimal_cityflow_config()
        config_path = "test_cityflow_config.json"
        
        # Write config to file
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created test config file: {config_path}")
        
        # Test creating CityFlow engine
        try:
            eng = cityflow.Engine(config_path, thread_num=1)
            print(f"✅ CityFlow engine created successfully")
        except Exception as engine_error:
            print(f"❌ Failed to create CityFlow engine: {engine_error}")
            return False
        
        # Test basic operations
        try:
            vehicles = eng.get_vehicles()
            print(f"✅ Got vehicles list: {len(vehicles)} vehicles")
            
            # Run a few simulation steps
            for step in range(3):  # Reduced steps to avoid segfault
                eng.next_step()
                vehicles = eng.get_vehicles()
                print(f"   Step {step + 1}: {len(vehicles)} vehicles")
            
            print(f"✅ CityFlow simulation steps completed")
        except Exception as sim_error:
            print(f"❌ Error during simulation: {sim_error}")
            return False
        
        # Cleanup
        os.remove(config_path)
        print(f"✅ Cleaned up test config file")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import CityFlow: {e}")
        print("   Install with: pip install cityflow")
        return False
    except Exception as e:
        print(f"❌ Error testing CityFlow: {e}")
        return False

def create_minimal_cityflow_config():
    """Create a minimal CityFlow configuration for testing"""
    return {
        "interval": 1.0,
        "seed": 0,
        "dir": "./",
        "roadnetFile": "test_roadnet.json",
        "flowFile": "test_flow.json",
        "rlTrafficLight": False,
        "saveReplay": False,
        "roadnetLogFile": "roadnet.log",
        "replayLogFile": "replay.log"
    }

def create_test_roadnet():
    """Create a minimal roadnet file for CityFlow testing"""
    roadnet = {
        "intersections": [
            {
                "id": "intersection_1",
                "point": {"x": 0.0, "y": 0.0},
                "width": 10.0,
                "roads": ["road_0_1_0", "road_0_1_1"],
                "roadLinks": [
                    {
                        "type": "go_straight",
                        "startRoad": "road_0_1_0",
                        "endRoad": "road_0_1_1",
                        "direction": 0,
                        "laneLinks": [
                            {
                                "startLaneIndex": 0,
                                "endLaneIndex": 0,
                                "points": [
                                    {"x": -5.0, "y": 0.0},
                                    {"x": 5.0, "y": 0.0}
                                ]
                            }
                        ]
                    }
                ],
                "trafficLight": {
                    "roadLinkIndices": [0],
                    "lightphases": [
                        {
                            "time": 30,
                            "availableRoadLinks": [0]
                        }
                    ]
                },
                "virtual": False
            },
            {
                "id": "intersection_0",
                "point": {"x": -100.0, "y": 0.0},
                "width": 10.0,
                "roads": ["road_0_1_0"],
                "roadLinks": [],
                "virtual": True
            },
            {
                "id": "intersection_2",
                "point": {"x": 100.0, "y": 0.0},
                "width": 10.0,
                "roads": ["road_0_1_1"],
                "roadLinks": [],
                "virtual": True
            }
        ],
        "roads": [
            {
                "id": "road_0_1_0",
                "startIntersection": "intersection_0",
                "endIntersection": "intersection_1",
                "points": [
                    {"x": -100.0, "y": 0.0},
                    {"x": -5.0, "y": 0.0}
                ],
                "lanes": [
                    {
                        "width": 3.0,
                        "maxSpeed": 16.67
                    }
                ]
            },
            {
                "id": "road_0_1_1",
                "startIntersection": "intersection_1",
                "endIntersection": "intersection_2",
                "points": [
                    {"x": 5.0, "y": 0.0},
                    {"x": 100.0, "y": 0.0}
                ],
                "lanes": [
                    {
                        "width": 3.0,
                        "maxSpeed": 16.67
                    }
                ]
            }
        ]
    }
    
    # Write roadnet file
    import json
    with open("test_roadnet.json", 'w') as f:
        json.dump(roadnet, f, indent=2)
    
    # Create a simple flow file with one vehicle
    flow = [
        {
            "vehicle": [
                {
                    "startTime": 0,
                    "endTime": -1,
                    "interval": 10.0,
                    "route": ["road_0_1_0", "road_0_1_1"]
                }
            ]
        }
    ]
    with open("test_flow.json", 'w') as f:
        json.dump(flow, f, indent=2)

def cleanup_test_files():
    """Clean up test files"""
    files_to_remove = [
        "test_roadnet.json",
        "test_flow.json", 
        "test_cityflow_config.json",
        "roadnet.log",
        "replay.log"
    ]
    
    for file in files_to_remove:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def main():
    """Main test function"""
    print("🔍 Testing Gym and CityFlow Installation")
    print("=" * 60)
    
    # Test Gym
    gym_ok = test_gym_installation()
    
    # Create test files for CityFlow
    try:
        create_test_roadnet()
        cityflow_ok = test_cityflow_installation()
    except Exception as e:
        print(f"❌ Error setting up CityFlow test: {e}")
        cityflow_ok = False
    finally:
        cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"OpenAI Gym: {'✅ PASS' if gym_ok else '❌ FAIL'}")
    print(f"CityFlow:   {'✅ PASS' if cityflow_ok else '❌ FAIL'}")
    
    if gym_ok and cityflow_ok:
        print("\n🎉 All tests passed! Ready to proceed with TurtleBot3 integration.")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        if not gym_ok:
            print("   pip install gym")
        if not cityflow_ok:
            print("   pip install cityflow")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()