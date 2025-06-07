import requests
import json
import base64
import time
import os
import argparse
from typing import Dict, Any, List, Optional
from PIL import Image
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class CerebriumModelTester:
    """Comprehensive tester for Cerebrium deployed model"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.test_results = []
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {str(e)}")
    
    def predict_single_image(self, image_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to the image file
            verbose: Whether to print results
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Encode image
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Prepare request
            payload = {
                "image_base64": image_base64
            }
            
            # Make request
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/predict",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result['total_request_time'] = total_time
                
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"Image: {os.path.basename(image_path)}")
                    print(f"Predicted Class ID: {result.get('predicted_class', 'N/A')}")
                    print(f"Confidence: {result.get('confidence', 'N/A'):.4f}")
                    print(f"Inference Time: {result.get('inference_time', 'N/A'):.4f}s")
                    print(f"Total Request Time: {total_time:.4f}s")
                    
                    if 'top5_predictions' in result:
                        print("\nTop 5 Predictions:")
                        for i, pred in enumerate(result['top5_predictions'], 1):
                            print(f"  {i}. Class {pred['class']}: {pred['confidence']:.4f}")
                
                return result
            else:
                error_result = {
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'status': 'error',
                    'total_request_time': total_time
                }
                if verbose:
                    print(f"Error: {error_result['error']}")
                return error_result
                
        except Exception as e:
            error_result = {
                'error': str(e),
                'status': 'error',
                'total_request_time': 0
            }
            if verbose:
                print(f"Exception: {str(e)}")
            return error_result
    
    def health_check(self) -> Dict[str, Any]:
        """Run health check on deployed model"""
        try:
            payload = {"test_mode": True}
            
            response = requests.post(
                f"{self.api_url}/predict",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def load_test(self, image_path: str, num_requests: int = 10, concurrent_requests: int = 3) -> Dict[str, Any]:
        """
        Perform load testing on the deployed model
        
        Args:
            image_path: Path to test image
            num_requests: Total number of requests to make
            concurrent_requests: Number of concurrent requests
            
        Returns:
            Load test results
        """
        print(f"\n{'='*50}")
        print(f"LOAD TEST: {num_requests} requests with {concurrent_requests} concurrent")
        print(f"{'='*50}")
        
        if not os.path.exists(image_path):
            return {'error': f'Test image not found: {image_path}'}
        
        image_base64 = self.encode_image_to_base64(image_path)
        payload = {"image_base64": image_base64}
        
        def make_request(request_id):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                total_time = time.time() - start_time
                
                return {
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'response_time': total_time,
                    'response_size': len(response.content),
                    'error': None if response.status_code == 200 else response.text
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'status_code': 0,
                    'success': False,
                    'response_time': time.time() - start_time,
                    'response_size': 0,
                    'error': str(e)
                }
        
        # Execute load test
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                status = "✓" if result['success'] else "✗"
                print(f"Request {result['request_id']:2d}: {status} "
                      f"{result['response_time']:.3f}s "
                      f"(HTTP {result['status_code']})")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r['success']]
        response_times = [r['response_time'] for r in successful_requests]
        
        load_test_results = {
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': num_requests - len(successful_requests),
            'success_rate': len(successful_requests) / num_requests * 100,
            'total_time': total_time,
            'requests_per_second': num_requests / total_time,
            'response_times': {
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'mean': statistics.mean(response_times) if response_times else 0,
                'median': statistics.median(response_times) if response_times else 0,
                'p95': np.percentile(response_times, 95) if response_times else 0,
                'p99': np.percentile(response_times, 99) if response_times else 0
            },
            'errors': [r['error'] for r in results if not r['success']]
        }
        
        # Print summary
        print(f"\nLOAD TEST SUMMARY:")
        print(f"Success Rate: {load_test_results['success_rate']:.1f}%")
        print(f"Requests/sec: {load_test_results['requests_per_second']:.2f}")
        print(f"Response Times (ms):")
        print(f"  Min: {load_test_results['response_times']['min']*1000:.1f}")
        print(f"  Mean: {load_test_results['response_times']['mean']*1000:.1f}")
        print(f"  Median: {load_test_results['response_times']['median']*1000:.1f}")
        print(f"  P95: {load_test_results['response_times']['p95']*1000:.1f}")
        print(f"  P99: {load_test_results['response_times']['p99']*1000:.1f}")
        print(f"  Max: {load_test_results['response_times']['max']*1000:.1f}")
        
        return load_test_results
    
    def availability_test(self, duration_minutes: int = 5, interval_seconds: int = 30) -> Dict[str, Any]:
        """
        Test model availability over time
        
        Args:
            duration_minutes: How long to run the test
            interval_seconds: Interval between health checks
            
        Returns:
            Availability test results
        """
        print(f"\n{'='*50}")
        print(f"AVAILABILITY TEST: {duration_minutes} minutes, {interval_seconds}s intervals")
        print(f"{'='*50}")
        
        end_time = time.time() + (duration_minutes * 60)
        checks = []
        
        while time.time() < end_time:
            start_time = time.time()
            health_result = self.health_check()
            check_time = time.time() - start_time
            
            is_healthy = health_result.get('status') == 'healthy'
            checks.append({
                'timestamp': start_time,
                'healthy': is_healthy,
                'response_time': check_time,
                'details': health_result
            })
            
            status = "✓" if is_healthy else "✗"
            print(f"{time.strftime('%H:%M:%S')}: {status} ({check_time:.3f}s)")
            
            time.sleep(interval_seconds)
        
        # Calculate availability metrics
        total_checks = len(checks)
        healthy_checks = sum(1 for c in checks if c['healthy'])
        availability_percentage = (healthy_checks / total_checks * 100) if total_checks > 0 else 0
        
        avg_response_time = statistics.mean([c['response_time'] for c in checks])
        
        availability_results = {
            'duration_minutes': duration_minutes,
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'availability_percentage': availability_percentage,
            'average_response_time': avg_response_time,
            'checks': checks
        }
        
        print(f"\nAVAILABILITY SUMMARY:")
        print(f"Availability: {availability_percentage:.2f}%")
        print(f"Average Response Time: {avg_response_time*1000:.1f}ms")
        
        return availability_results
    
    def run_comprehensive_tests(self, test_image_path: str) -> Dict[str, Any]:
        """Run all tests in sequence"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE CEREBRIUM DEPLOYMENT TESTS")
        print(f"{'='*60}")
        
        all_results = {
            'timestamp': time.time(),
            'test_image': test_image_path
        }
        
        # 1. Health Check
        print("\n1. HEALTH CHECK")
        print("-" * 30)
        health_result = self.health_check()
        all_results['health_check'] = health_result
        
        if health_result.get('status') == 'healthy':
            print("✓ Model is healthy")
            for check_name, check_result in health_result.get('checks', {}).items():
                status = check_result.get('status', 'unknown')
                print(f"  {check_name}: {status}")
        else:
            print("✗ Model is unhealthy")
            print(f"  Error: {health_result.get('error', 'Unknown error')}")
        
        # 2. Single Prediction Test
        print("\n2. SINGLE PREDICTION TEST")
        print("-" * 30)
        prediction_result = self.predict_single_image(test_image_path, verbose=True)
        all_results['single_prediction'] = prediction_result
        
        # 3. Load Test
        print("\n3. LOAD TEST")
        print("-" * 30)
        load_result = self.load_test(test_image_path, num_requests=20, concurrent_requests=5)
        all_results['load_test'] = load_result
        
        # 4. Availability Test (shorter for demo)
        print("\n4. AVAILABILITY TEST")
        print("-" * 30)
        availability_result = self.availability_test(duration_minutes=2, interval_seconds=15)
        all_results['availability_test'] = availability_result
        
        # 5. Error Handling Test
        print("\n5. ERROR HANDLING TEST")
        print("-" * 30)
        error_tests = self.test_error_handling()
        all_results['error_handling'] = error_tests
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        print(f"Health Status: {'✓' if health_result.get('status') == 'healthy' else '✗'}")
        print(f"Prediction Success: {'✓' if prediction_result.get('status') == 'success' else '✗'}")
        print(f"Load Test Success Rate: {load_result.get('success_rate', 0):.1f}%")
        print(f"Availability: {availability_result.get('availability_percentage', 0):.1f}%")
        
        return all_results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test various error conditions"""
        error_tests = {}
        
        # Test with invalid base64
        try:
            payload = {"image_base64": "invalid_base64_string"}
            response = requests.post(f"{self.api_url}/predict", headers=self.headers, json=payload, timeout=10)
            error_tests['invalid_base64'] = {
                'status_code': response.status_code,
                'handled_gracefully': response.status_code != 500
            }
        except Exception as e:
            error_tests['invalid_base64'] = {'error': str(e), 'handled_gracefully': False}
        
        # Test with empty payload
        try:
            response = requests.post(f"{self.api_url}/predict", headers=self.headers, json={}, timeout=10)
            error_tests['empty_payload'] = {
                'status_code': response.status_code,
                'handled_gracefully': response.status_code != 500
            }
        except Exception as e:
            error_tests['empty_payload'] = {'error': str(e), 'handled_gracefully': False}
        
        # Test with malformed JSON
        try:
            response = requests.post(
                f"{self.api_url}/predict", 
                headers=self.headers, 
                data="invalid json", 
                timeout=10
            )
            error_tests['malformed_json'] = {
                'status_code': response.status_code,
                'handled_gracefully': response.status_code != 500
            }
        except Exception as e:
            error_tests['malformed_json'] = {'error': str(e), 'handled_gracefully': False}
        
        print("Error Handling Tests:")
        for test_name, result in error_tests.items():
            handled = result.get('handled_gracefully', False)
            print(f"  {test_name}: {'✓' if handled else '✗'}")
        
        return error_tests

def main():
    parser = argparse.ArgumentParser(description='Test Cerebrium deployed model')
    parser.add_argument('--api-url', required=True, help='Cerebrium API URL')
    parser.add_argument('--api-key', required=True, help='Cerebrium API key')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive tests')
    parser.add_argument('--load-test', action='store_true', help='Run load test only')
    parser.add_argument('--health-check', action='store_true', help='Run health check only')
    parser.add_argument('--availability-test', type=int, help='Run availability test for N minutes')
    
    args = parser.parse_args()
    
    tester = CerebriumModelTester(args.api_url, args.api_key)
    
    if args.comprehensive:
        if not args.image:
            print("Error: --image required for comprehensive tests")
            return
        results = tester.run_comprehensive_tests(args.image)
        
        # Save results to file
        with open('cerebrium_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to cerebrium_test_results.json")
        
    elif args.health_check:
        result = tester.health_check()
        print(json.dumps(result, indent=2))
        
    elif args.load_test:
        if not args.image:
            print("Error: --image required for load test")
            return
        result = tester.load_test(args.image)
        print(json.dumps(result, indent=2, default=str))
        
    elif args.availability_test:
        result = tester.availability_test(duration_minutes=args.availability_test)
        print(json.dumps(result, indent=2, default=str))
        
    elif args.image:
        result = tester.predict_single_image(args.image)
        print(f"\nPredicted Class ID: {result.get('predicted_class', 'Error')}")
        
    else:
        print("Please specify an action: --image, --comprehensive, --load-test, --health-check, or --availability-test")

if __name__ == "__main__":
    main()