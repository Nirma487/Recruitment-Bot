import cv2
import mediapipe as mp
import numpy as np
import time
import json
from datetime import datetime
import webbrowser
import os

class InterviewAssessment:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.scores = {
            'body_language': [],
            'posture': [],
            'active_listening': [],
            'attitude': [],
            'communication': []
        }
        self.frame_count = 0
        self.start_time = time.time()
        
    def analyze_posture(self, landmarks):
        if landmarks.pose_landmarks:
            # Check shoulder alignment
            left_shoulder = landmarks.pose_landmarks.landmark[11]
            right_shoulder = landmarks.pose_landmarks.landmark[12]
            shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
            
            # Check spine alignment
            nose = landmarks.pose_landmarks.landmark[0]
            mid_hip = landmarks.pose_landmarks.landmark[24]
            spine_alignment = abs(nose.x - mid_hip.x)
            
            posture_score = 10 * (1 - (shoulder_alignment + spine_alignment) / 2)
            return max(0, min(10, posture_score))
        return 5

    def analyze_body_language(self, landmarks):
        if landmarks.pose_landmarks and landmarks.face_landmarks:
            # Analyze hand movements
            hand_movement = 0
            if landmarks.left_hand_landmarks or landmarks.right_hand_landmarks:
                hand_movement = 5
            
            # Analyze facial expressions using face landmarks
            face_score = 5
            if landmarks.face_landmarks:
                face_score = 7
                
            return (hand_movement + face_score) / 2
        return 5

    def analyze_active_listening(self, landmarks):
        if landmarks.face_landmarks:
            # Analyze head position and movement
            head_pos = landmarks.face_landmarks.landmark[0]  # Nose tip
            self.frame_count += 1
            
            # Simple head movement detection
            if self.frame_count > 1:
                movement = abs(head_pos.x - self.prev_head_pos) + abs(head_pos.y - self.prev_head_pos)
                score = 7 + min(3, movement * 100)  # Reward slight movement
                self.prev_head_pos = head_pos.x
                return max(0, min(10, score))
            
            self.prev_head_pos = head_pos.x
        return 5

    def analyze_attitude(self, landmarks):
        if landmarks.face_landmarks and landmarks.pose_landmarks:
            # Combine facial expressions and body posture
            facial_score = self.analyze_facial_expression(landmarks.face_landmarks)
            posture_score = self.analyze_posture(landmarks)
            return (facial_score + posture_score) / 2
        return 5

    def analyze_facial_expression(self, face_landmarks):
        if face_landmarks:
            # Basic facial expression analysis
            return 7 + np.random.normal(0, 0.5)  # Simplified for demo
        return 5

    def analyze_communication(self, landmarks):
        if landmarks.face_landmarks and (landmarks.left_hand_landmarks or landmarks.right_hand_landmarks):
            # Analyze facial movements and hand gestures
            face_movement = self.analyze_facial_expression(landmarks.face_landmarks)
            gesture_score = 8 if landmarks.left_hand_landmarks or landmarks.right_hand_landmarks else 5
            return (face_movement + gesture_score) / 2
        return 5

    def save_results(self):
        # Calculate average scores
        final_scores = {k: np.mean(v) if v else 5.0 for k, v in self.scores.items()}
        
        # Generate feedback
        feedback = {
            'body_language': {
                'score': final_scores['body_language'],
                'strengths': 'Good use of hand gestures' if final_scores['body_language'] > 7 else '',
                'improvements': 'Try to use more hand gestures' if final_scores['body_language'] < 7 else ''
            },
            'posture': {
                'score': final_scores['posture'],
                'strengths': 'Excellent posture maintained' if final_scores['posture'] > 7 else '',
                'improvements': 'Try to maintain a straighter posture' if final_scores['posture'] < 7 else ''
            },
            'active_listening': {
                'score': final_scores['active_listening'],
                'strengths': 'Great engagement shown' if final_scores['active_listening'] > 7 else '',
                'improvements': 'Show more engagement through nodding' if final_scores['active_listening'] < 7 else ''
            },
            'attitude': {
                'score': final_scores['attitude'],
                'strengths': 'Positive attitude displayed' if final_scores['attitude'] > 7 else '',
                'improvements': 'Try to show more enthusiasm' if final_scores['attitude'] < 7 else ''
            },
            'communication': {
                'score': final_scores['communication'],
                'strengths': 'Clear and effective communication' if final_scores['communication'] > 7 else '',
                'improvements': 'Work on speaking clarity and pace' if final_scores['communication'] < 7 else ''
            }
        }
        
        # Save results to a file
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scores': final_scores,
            'feedback': feedback
        }
        
        with open('interview_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        # Create and open results HTML page
        self.create_results_page(results)
        webbrowser.open('file://' + os.path.realpath('interview_results.html'))

    def create_results_page(self, results):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interview Assessment Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .score-card {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .score {{
                    font-size: 24px;
                    color: #2196F3;
                }}
                .feedback {{
                    margin-top: 10px;
                }}
                .strengths {{
                    color: #4CAF50;
                }}
                .improvements {{
                    color: #FF5722;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <h1>Interview Assessment Results</h1>
            <p style="text-align: center">Assessment completed on: {results['timestamp']}</p>
        """
        
        for category, data in results['feedback'].items():
            html_content += f"""
            <div class="score-card">
                <h2>{category.replace('_', ' ').title()}</h2>
                <div class="score">Score: {data['score']:.1f}/10</div>
                <div class="feedback">
                    <p class="strengths">{data['strengths']}</p>
                    <p class="improvements">{data['improvements']}</p>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open('interview_results.html', 'w') as f:
            f.write(html_content)

    def start_assessment(self):
        cap = cv2.VideoCapture(0)
        
        # Improve video quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Improve frame quality
                frame = cv2.flip(frame, 1)  # Mirror image
                frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Reduce noise
                
                # Convert to RGB for MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks with improved visibility
                self.mp_drawing.draw_landmarks(
                    image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Analyze metrics
                self.scores['posture'].append(self.analyze_posture(results))
                self.scores['body_language'].append(self.analyze_body_language(results))
                self.scores['active_listening'].append(self.analyze_active_listening(results))
                self.scores['attitude'].append(self.analyze_attitude(results))
                self.scores['communication'].append(self.analyze_communication(results))

                # Display current scores
                y_pos = 30
                for metric, scores in self.scores.items():
                    current_score = np.mean(scores[-10:]) if scores else 0
                    cv2.putText(image, f"{metric.replace('_', ' ').title()}: {current_score:.1f}",
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 30

                # Show remaining time
                elapsed_time = time.time() - self.start_time
                remaining_time = max(0, 60 - elapsed_time)
                cv2.putText(image, f"Time remaining: {int(remaining_time)}s",
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow('Interview Assessment', image)

                if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= 60:
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.save_results()

if __name__ == "__main__":
    assessment = InterviewAssessment()
    assessment.start_assessment()
