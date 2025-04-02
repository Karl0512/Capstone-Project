import os
import sys

def main():
    while True:
        print("Select an option:")
        print("1. Enroll Face (Save Face)")
        print("2. Run Face Recognition")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            # Run the face enrollment (saving faces) script
            os.system('python save_face.py')
        elif choice == '2':
            # Run the face recognition script
            os.system('python face_recog.py')
        elif choice == '3':
            print("Exiting program.")
            sys.exit(0)
        else:
            print("Invalid choice, please try again.")

if __name__ == '__main__':
    main()
