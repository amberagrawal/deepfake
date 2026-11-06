import os
import sys
import csv
import cv2

from video_processor import open_video
from face_detector import FaceDetector

def main():

    video_path = input("Enter the complete path of your video: ").strip().strip('"')

    if not os.path.isfile(video_path):
        print("ERROR: Video file does not exist.")
        return

    os.makedirs("outputs/landmarks", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    cap, fps, width, height, total_frames = open_video(video_path)

    output_video = "outputs/face_landmarks_output.mp4"
    output_csv = "outputs/landmarks/landmarks.csv"

    fps = fps if fps and fps > 0 else 25

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    detector = FaceDetector()

    frame_number = 0
    detected_frames = 0

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame",
                "landmark_index",
                "x",
                "y",
                "z"
            ])

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                result = detector.detect(frame)

                if result is not None:
                    face_landmarks, coordinates = result

                    detected_frames += 1

                    detector.draw(frame, face_landmarks)

                    for index, point in enumerate(coordinates):
                        csv_writer.writerow([
                            frame_number,
                            index,
                            point[0],
                            point[1],
                            point[2]
                        ])

                    cv2.putText(
                        frame,
                        "468 Facial Landmarks Detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                else:
                    cv2.putText(
                        frame,
                        "No Face Detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                writer.write(frame)

                if frame_number % 30 == 0:
                    print(
                        f"Processed {frame_number}/{total_frames} frames"
                    )

    finally:
        cap.release()
        writer.release()
        detector.close()

    print("\n========== PROCESSING COMPLETE ==========")
    print(f"Total frames processed: {frame_number}")
    print(f"Frames with detected face: {detected_frames}")

    if frame_number > 0:
        percentage = detected_frames / frame_number * 100
        print(f"Face detection rate: {percentage:.2f}%")

    print(f"Output video: {output_video}")
    print(f"Landmark coordinates: {output_csv}")

    if detected_frames == 0:
        print("WARNING: No faces were detected. Try another video.")

if __name__ == "__main__":
    main()
