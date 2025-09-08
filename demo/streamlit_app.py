#!/usr/bin/env python3
"""
Streamlit Web Demo cho Vietnam Traffic AI System
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import json
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules.object_detection import ObjectDetector
from modules.traffic_sign_detection import TrafficSignDetector
from modules.pose_analysis import PoseAnalyzer

# Page config
st.set_page_config(
    page_title="Vietnam Traffic AI System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 2rem;
}
.module-header {
    font-size: 1.5rem;
    color: #4ECDC4;
    margin-bottom: 1rem;
}
.stats-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load AI models (cached)"""
    config_path = "configs/config.yaml"
    
    try:
        object_detector = ObjectDetector(config_path)
        sign_detector = TrafficSignDetector(config_path)  
        pose_analyzer = PoseAnalyzer(config_path)
        
        return object_detector, sign_detector, pose_analyzer
    except Exception as e:
        st.error(f"Lỗi khi tải models: {str(e)}")
        return None, None, None

def process_image(image, modules, enable_modules):
    """Process image with selected modules"""
    results = {}
    processed_image = image.copy()
    
    try:
        # Object Detection
        if enable_modules.get('objects', False) and modules[0] is not None:
            detections = modules[0].detect(image)
            processed_image = modules[0].draw_detections(processed_image, detections)
            results['objects'] = modules[0].get_traffic_statistics(detections)
        
        # Traffic Sign Detection
        if enable_modules.get('signs', False) and modules[1] is not None:
            signs = modules[1].detect_traffic_signs(image)
            processed_image = modules[1].draw_signs(processed_image, signs)
            results['signs'] = modules[1].get_sign_statistics(signs)
        
        # Pose Analysis
        if enable_modules.get('pose', False) and modules[2] is not None:
            poses = modules[2].analyze_pose(image)
            processed_image = modules[2].draw_pose(processed_image, poses)
            results['poses'] = modules[2].get_pose_statistics(poses)
        
        return processed_image, results
    
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
        return image, {}

def main():
    # Header
    st.markdown('<h1 class="main-header">🚦 Vietnam Traffic AI System</h1>', unsafe_allow_html=True)
    st.markdown("### Hệ thống AI phân tích giao thông Việt Nam")
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Cấu hình")
    
    # Module selection
    st.sidebar.markdown("### Chọn Module AI")
    enable_objects = st.sidebar.checkbox("🚗 Phát hiện đối tượng", value=True)
    enable_signs = st.sidebar.checkbox("🚸 Nhận dạng biển báo", value=True)  
    enable_pose = st.sidebar.checkbox("🚶 Phân tích tư thế", value=True)
    
    enable_modules = {
        'objects': enable_objects,
        'signs': enable_signs, 
        'pose': enable_pose
    }
    
    # Load models
    with st.sidebar:
        if st.button("🔄 Tải lại Models"):
            st.cache_resource.clear()
        
        with st.spinner("Đang tải AI models..."):
            modules = load_models()
    
    if modules[0] is None:
        st.error("❌ Không thể tải models. Vui lòng kiểm tra cấu hình.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Upload Ảnh", "📹 Upload Video", "📊 Thống kê", "ℹ️ Hướng dẫn"])
    
    with tab1:
        st.markdown('<div class="module-header">Upload và phân tích ảnh</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh giao thông...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload ảnh giao thông để phân tích"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 Ảnh gốc")
                st.image(image, use_column_width=True)
            
            # Process image
            with st.spinner("Đang phân tích..."):
                processed_image, results = process_image(image_cv, modules, enable_modules)
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("#### 🔍 Ảnh đã phân tích")
                st.image(processed_image_rgb, use_column_width=True)
            
            # Display results
            if results:
                st.markdown("### 📊 Kết quả phân tích")
                
                col1, col2, col3 = st.columns(3)
                
                # Object detection results
                if 'objects' in results:
                    with col1:
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        st.markdown("#### 🚗 Đối tượng")
                        obj_stats = results['objects']
                        st.metric("Tổng đối tượng", obj_stats.get('total_objects', 0))
                        st.metric("Phương tiện", obj_stats.get('vehicles', 0))
                        st.metric("Người đi bộ", obj_stats.get('people', 0))
                        st.metric("Xe máy", obj_stats.get('motorcycles', 0))
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Traffic sign results  
                if 'signs' in results:
                    with col2:
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        st.markdown("#### 🚸 Biển báo")
                        sign_stats = results['signs']
                        st.metric("Tổng biển báo", sign_stats.get('total_signs', 0))
                        st.metric("Độ tin cậy cao", sign_stats.get('high_confidence', 0))
                        
                        if sign_stats.get('by_type'):
                            st.markdown("**Loại biển báo:**")
                            for sign_type, count in sign_stats['by_type'].items():
                                st.text(f"• {sign_type}: {count}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Pose analysis results
                if 'poses' in results:
                    with col3:
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        st.markdown("#### 🚶 Tư thế người")
                        pose_stats = results['poses'] 
                        st.metric("Người phát hiện", pose_stats.get('total_people', 0))
                        
                        if pose_stats.get('risk_levels'):
                            st.markdown("**Mức độ rủi ro:**")
                            for level, count in pose_stats['risk_levels'].items():
                                st.text(f"• {level}: {count}")
                        
                        if pose_stats.get('behaviors'):
                            st.markdown("**Hành vi:**")
                            for behavior, count in pose_stats['behaviors'].items():
                                st.text(f"• {behavior}: {count}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Download processed image
                st.markdown("### 💾 Tải xuống")
                processed_pil = Image.fromarray(processed_image_rgb)
                
                # Convert to bytes
                import io
                buf = io.BytesIO()
                processed_pil.save(buf, format='PNG')
                
                st.download_button(
                    label="📥 Tải ảnh đã xử lý",
                    data=buf.getvalue(),
                    file_name=f"processed_{uploaded_file.name}",
                    mime="image/png"
                )
    
    with tab2:
        st.markdown('<div class="module-header">Upload và phân tích video</div>', unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader(
            "Chọn video giao thông...",
            type=['mp4', 'avi', 'mov'],
            help="Upload video giao thông để phân tích"
        )
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                temp_video_path = tmp_file.name
            
            # Display video
            st.video(uploaded_video)
            
            if st.button("🔍 Phân tích Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video frame by frame
                cap = cv2.VideoCapture(temp_video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Prepare output video
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_out:
                    output_path = tmp_out.name
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                video_stats = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame, frame_results = process_image(frame, modules, enable_modules)
                    out.write(processed_frame)
                    
                    # Collect stats
                    video_stats.append(frame_results)
                    
                    # Update progress
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Xử lý frame {frame_count}/{total_frames}")
                
                cap.release()
                out.release()
                
                st.success("✅ Video đã được xử lý!")
                
                # Display processed video
                with open(output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Download processed video
                st.download_button(
                    label="📥 Tải video đã xử lý",
                    data=video_bytes,
                    file_name=f"processed_{uploaded_video.name}",
                    mime="video/mp4"
                )
                
                # Cleanup temp files
                os.unlink(temp_video_path)
                os.unlink(output_path)
    
    with tab3:
        st.markdown('<div class="module-header">Thống kê và phân tích</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📈 Dashboard thống kê
        Khu vực này sẽ hiển thị:
        - Biểu đồ thống kê theo thời gian
        - Phân tích mật độ giao thông
        - Xu hướng các loại phương tiện
        - Tần suất biển báo
        - Phân tích hành vi người đi bộ
        """)
        
        # Placeholder for statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🚗 Thống kê phương tiện")
            # Create sample chart
            import pandas as pd
            import matplotlib.pyplot as plt
            
            sample_data = {
                'Loại xe': ['Xe máy', 'Ô tô', 'Xe buýt', 'Xe tải', 'Xe đạp'],
                'Số lượng': [150, 80, 15, 25, 30]
            }
            df = pd.DataFrame(sample_data)
            st.bar_chart(df.set_index('Loại xe'))
        
        with col2:
            st.markdown("#### 🚸 Thống kê biển báo")
            sample_signs = {
                'Loại biển': ['Cấm', 'Chỉ dẫn', 'Cảnh báo', 'Giới hạn tốc độ'],
                'Số lượng': [25, 40, 30, 20]
            }
            df_signs = pd.DataFrame(sample_signs)
            st.bar_chart(df_signs.set_index('Loại biển'))
    
    with tab4:
        st.markdown('<div class="module-header">Hướng dẫn sử dụng</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 Giới thiệu
        **Vietnam Traffic AI System** là hệ thống AI tích hợp nhiều module để phân tích giao thông Việt Nam.
        
        ## 🔧 Các tính năng chính
        
        ### 🚗 Phát hiện đối tượng
        - Nhận dạng xe cộ (ô tô, xe máy, xe buýt, xe tải)
        - Phát hiện người đi bộ
        - Thống kê mật độ giao thông
        - Đặc biệt tối ưu cho xe máy (đặc trưng VN)
        
        ### 🚸 Nhận dạng biển báo
        - Đọc biển báo tiếng Việt và tiếng Anh
        - Phân loại các loại biển báo giao thông
        - Phát hiện biển tốc độ, cấm, chỉ dẫn
        - Xử lý trong điều kiện thời tiết khác nhau
        
        ### 🚶 Phân tích tư thế
        - Theo dõi hướng nhìn của người đi bộ
        - Phát hiện cử chỉ ra hiệu
        - Đánh giá mức độ rủi ro
        - Phân tích hành vi giao thông
        
        ## 🚀 Cách sử dụng
        
        1. **Upload ảnh/video**: Chọn file từ máy tính
        2. **Chọn module**: Bật/tắt các tính năng theo nhu cầu
        3. **Phân tích**: Nhấn nút phân tích và chờ kết quả
        4. **Xem kết quả**: Quan sát ảnh/video đã xử lý và thống kê
        5. **Tải xuống**: Download kết quả về máy
        
        ## ⚙️ Cấu hình
        
        - **Độ tin cậy**: Điều chỉnh ngưỡng phát hiện
        - **Chất lượng**: Chọn giữa tốc độ và độ chính xác
        - **Module**: Bật/tắt các tính năng theo nhu cầu
        
        ## 🔄 Command Line
        
        Ngoài web interface, bạn có thể sử dụng qua command line:
        
        ```bash
        # Webcam real-time
        python main.py --mode webcam
        
        # Xử lý video
        python main.py --mode video --input video.mp4 --output result.mp4
        
        # Xử lý batch images
        python main.py --mode images --input images/ --output processed/
        
        # Phân tích thống kê
        python main.py --mode analyze --input video.mp4 --output stats.json
        ```
        
        ## 📞 Hỗ trợ
        
        - **GitHub**: [vietnam-traffic-ai](https://github.com/your-repo)
        - **Issues**: Báo cáo lỗi và góp ý
        - **Documentation**: Tài liệu chi tiết trong thư mục `docs/`
        """)
        
        # System info
        st.markdown("---")
        st.markdown("### 🔍 Thông tin hệ thống")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Framework**: YOLO, MediaPipe, EasyOCR")
        with col2:
            st.info("**Language**: Python 3.8+")
        with col3:
            st.info("**Platform**: Cross-platform")

if __name__ == "__main__":
    main()