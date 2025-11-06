import cv2
import mediapipe as mp
import numpy as np

# --- กำหนดค่าเริ่มต้น ---

# ค่าสีสำหรับแปรงวาด (BGR)
blue = (255, 160, 0)
green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0) # สีดำสำหรับยางลบ (ต้องตรงกับสีพื้นหลังของ Canvas)
white = (255, 255, 255)

# สีที่ถูกเลือกเริ่มต้น
draw_color = blue
# ความหนาของเส้น
brush_thickness = 8
eraser_thickness = 35 # ความหนาสำหรับยางลบ

# ขนาดหน้าจอ
W, H = 640, 480

# --- การปรับเส้นให้สมูท (Smoothing) ---
SMOOTHING_ALPHA = 0.5 # ค่า Alpha ยิ่งน้อยยิ่งสมูท (แนะนำ 0.5 - 0.8)
# State สำหรับเก็บพิกัดที่ถูกปรับให้สมูทแล้ว (ตัวแปรเหล่านี้อยู่ใน Global Scope)
x_smooth, y_smooth = W // 2, H // 2 

# สร้างพื้นที่วาดรูป (Canvas) สีดำ
canvas = np.zeros((H, W, 3), dtype=np.uint8)

# ตัวแปรสำหรับเก็บพิกัดของเส้นที่วาด (จุดเริ่มต้นของเส้นวาดครั้งสุดท้าย)
pX, pY = 0, 0

# กำหนดโหมดการตรวจจับมือของ MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)
# mp_draw = mp.solutions.drawing_utils # ไม่ได้ใช้สำหรับวาดเส้นมือ

# --- การทำงานของกล้องและลูปหลัก ---

cap = cv2.VideoCapture(0)
# ปรับขนาดเฟรม
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

print("--- พร้อมใช้งานแล้ว! (Air Painter) ---")
print("1. วาด: นิ้วชี้เดี่ยวชี้ขึ้น")
print("2. เลือก/ควบคุม: นิ้วชี้และนิ้วกลางชี้ขึ้น")
print("กด 'q' เพื่อออก")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    # 1. พลิกภาพ (Mirror)
    img = cv2.flip(img, 1)

    # 2. ประมวลผลภาพสำหรับ MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # 3. ตรวจสอบการตรวจจับมือ
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # ดึงพิกัดของ Landmark ทั้งหมด
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                # แปลงค่าพิกัดให้เป็นพิกเซล
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if lmList:
                # พิกัดของปลาย 'นิ้วชี้' (x1, y1) และ 'นิ้วกลาง' (x2, y2)
                x1, y1 = lmList[8][1:] # นิ้วชี้ (Index Finger Tip)
                x2, y2 = lmList[12][1:] # นิ้วกลาง (Middle Finger Tip)

                # --- 1. การทำเส้นให้สมูทด้วย Exponential Moving Average (EMA) ---
                if pX == 0 and pY == 0:
                    x_smooth, y_smooth = x1, y1
                else:
                    # EMA: New_Smoothed = Alpha * Raw + (1 - Alpha) * Old_Smoothed
                    x_smooth = int(SMOOTHING_ALPHA * x1 + (1 - SMOOTHING_ALPHA) * x_smooth)
                    y_smooth = int(SMOOTHING_ALPHA * y1 + (1 - SMOOTHING_ALPHA) * y_smooth)
                
                # ใช้ค่าที่ smooth แล้วสำหรับ logic และการวาด
                x_active, y_active = x_smooth, y_smooth
                
                # --- ตรวจสอบสถานะนิ้วมือ (Improved Logic) ---
                
                # Check 1: นิ้วชี้ขึ้น (Index Finger Tip (8) ต้องอยู่สูงกว่า PIP Joint (6))
                index_up = lmList[8][2] < lmList[6][2]
                
                # Check 2: นิ้วกลาง, นิ้วนาง, นิ้วก้อย พับลง (Tip ต้องอยู่ต่ำกว่า PIP Joint)
                middle_down = lmList[12][2] > lmList[10][2]
                ring_down = lmList[16][2] > lmList[14][2]
                pinky_down = lmList[20][2] > lmList[18][2]

                # โหมดวาดรูป: นิ้วชี้ขึ้นนิ้วเดียว
                is_drawing_mode = index_up and middle_down and ring_down and pinky_down
                
                # โหมดเลือก/ควบคุม: นิ้วชี้และนิ้วกลางขึ้นสองนิ้ว
                is_selection_mode = lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2]

                # วาดวงกลมที่ปลายนิ้วชี้เสมอ เพื่อบอกตำแหน่ง
                current_thickness = eraser_thickness if draw_color == black else brush_thickness
                cv2.circle(img, (x1, y1), current_thickness, draw_color, cv2.FILLED)

                # --- A. โหมดเลือกสี/ควบคุม (สองนิ้วชี้ขึ้น) ---
                if is_selection_mode:
                    
                    # รีเซ็ตจุดเริ่มต้นเมื่อเปลี่ยนโหมด
                    pX, pY = 0, 0
                    
                    # แถบควบคุมอยู่ด้านบน (y_active น้อยกว่า 60)
                    if y_active < 60:
                        
                        # กำหนดโซนควบคุม (แถบ 640x60 ถูกแบ่งเป็น 5 ส่วนเท่าๆ กัน)
                        zone_width = W // 5
                        
                        # 1. BLUE (0 - 128)
                        if 0 < x_active < zone_width:
                            draw_color = blue
                            brush_thickness = 8
                            print("Selected: Blue")
                            
                        # 2. GREEN (128 - 256)
                        elif zone_width < x_active < zone_width * 2:
                            draw_color = green
                            brush_thickness = 8
                            print("Selected: Green")
                            
                        # 3. RED (256 - 384)
                        elif zone_width * 2 < x_active < zone_width * 3:
                            draw_color = red
                            brush_thickness = 8
                            print("Selected: Red")
                            
                        # 4. ERASER (384 - 512)
                        elif zone_width * 3 < x_active < zone_width * 4:
                            draw_color = black # ตั้งค่าสีเป็นสีดำ (สีพื้นหลัง)
                            brush_thickness = eraser_thickness # เพิ่มความหนา
                            print("Selected: Eraser")

                        # 5. CLEAR (512 - 640)
                        elif zone_width * 4 < x_active < W:
                            canvas = np.zeros((H, W, 3), dtype=np.uint8)
                            print("Action: Clear Canvas")
                            # หลังเคลียร์ ให้กลับไปใช้สีวาดล่าสุด (ไม่ใช่ยางลบ)
                            if draw_color == black:
                                draw_color = blue
                                brush_thickness = 8
                                
                    # แสดงข้อความ "SELECT MODE"
                    cv2.putText(img, "Selection Mode", (W // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)
                
                # --- B. โหมดวาดรูป (นิ้วชี้เดี่ยวชี้ขึ้น) ---
                elif is_drawing_mode:
                    
                    # แสดงข้อความ "DRAW MODE"
                    cv2.putText(img, "Draw Mode", (W // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                    
                    # ถ้าปลายนิ้วชี้อยู่ในแถบควบคุมด้านบน จะไม่ทำการวาด
                    if y_active < 60:
                        pX, pY = 0, 0 # หยุดวาด
                        
                    # ถ้าไม่ได้อยู่ในแถบควบคุม ให้ทำการวาด
                    else:
                        if pX == 0 and pY == 0:
                            # กำหนดจุดเริ่มต้นของเส้นวาดใหม่
                            pX, pY = x_active, y_active
                        
                        # วาดเส้นบน Canvas (จุดต่อจุด)
                        current_thickness = eraser_thickness if draw_color == black else brush_thickness
                        # ใช้ค่าที่สมูทแล้ว (x_active, y_active) ในการวาด เพื่อให้เส้นต่อเนื่อง
                        cv2.line(canvas, (pX, pY), (x_active, y_active), draw_color, current_thickness)
                        
                        # อัพเดตจุดเริ่มต้นใหม่
                        pX, pY = x_active, y_active
                        
                # --- C. โหมดหยุด (กำมือหรือมือลง) ---
                else:
                    # เมื่อหยุดวาด ให้รีเซ็ตจุดเริ่มต้น
                    pX, pY = 0, 0
                    
    # 4. รวมภาพ (Overlay) Canvas เข้ากับวิดีโอ
    
    # แปลง Canvas ที่เป็นสีดำ/เส้นวาด ให้เหลือเฉพาะเส้นวาด
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # 50 คือค่า Threshold สำหรับแยกสีดำกับสีอื่น
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV) 
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    # ผสมภาพวิดีโอ (img) กับส่วนที่วาดแล้ว (canvas)
    # หลักการ: (วิดีโอ AND InvertCanvas) + (Canvas AND Canvas)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)
    
    # 5. แสดงผลลัพธ์
    
    # เพิ่มแถบควบคุมที่ด้านบน
    # พื้นหลังสีเทาเข้ม
    cv2.rectangle(img, (0, 0), (W, 60), (50, 50, 50), -1) 
    
    # โซนควบคุมถูกแบ่งเป็น 5 ส่วน (128 พิกเซลต่อโซน)
    zone_width = W // 5
    
    # Draw Separators
    for i in range(1, 5):
        cv2.line(img, (zone_width * i, 0), (zone_width * i, 60), white, 1)

    # 1. BLUE
    cv2.putText(img, "BLUE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
    
    # 2. GREEN
    cv2.putText(img, "GREEN", (zone_width + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
    
    # 3. RED
    cv2.putText(img, "RED", (zone_width * 2 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    
    # 4. ERASER
    cv2.putText(img, "ERASER", (zone_width * 3 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
    
    # 5. CLEAR
    cv2.putText(img, "CLEAR", (zone_width * 4 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
    
    # แสดงสีปัจจุบัน/โหมดที่ใช้งาน
    cv2.circle(img, (W - 20, H - 20), 10, draw_color, cv2.FILLED)
    cv2.putText(img, "Current Color/Tool", (W - 250, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
    
    # แสดงภาพ
    cv2.imshow("Air Painter", img)

    # 6. ปุ่มออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดทุกอย่างเมื่อจบโปรแกรม
cap.release()
cv2.destroyAllWindows()