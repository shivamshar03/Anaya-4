
# Start the webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")
i=0
while i<5:
    i+=1
    ret, frame = cap.read()

    if not ret:
        break

    try:
        # Analyze the frame
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Extract dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        print(dominant_emotion)

        # Display the result
        cv2.putText(frame,
                    f'Emotion: {dominant_emotion}',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_4)

    except Exception as e:
        print(f"Error: {e}")



    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
