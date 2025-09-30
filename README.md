<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eye Tracking</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            cursor: none;
        }

        body {
            background: #ffffff;
            overflow: hidden;
        }

        #dot {
            position: fixed;
            width: 20px;
            height: 20px;
            background: #000;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 9999;
            will-change: transform;
        }

        #calibration {
            position: fixed;
            inset: 0;
            background: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        }

        #calibration.hidden {
            display: none;
        }

        .cal-point {
            position: absolute;
            width: 24px;
            height: 24px;
            background: #000;
            border-radius: 50%;
            transform: translate(-50%, -50%) scale(0);
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
            50% { transform: translate(-50%, -50%) scale(1.5); opacity: 0.6; }
        }

        .cal-point.active {
            transform: translate(-50%, -50%) scale(1);
        }

        #start-btn {
            background: #000;
            color: #fff;
            border: none;
            padding: 20px 40px;
            font-size: 18px;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.2s;
        }

        #start-btn:hover {
            transform: scale(1.05);
        }

        #video {
            position: fixed;
            top: -9999px;
            visibility: hidden;
        }
    </style>
</head>
<body>
    <div id="dot"></div>
    <video id="video" autoplay playsinline></video>

    <div id="calibration">
        <button id="start-btn">Начать</button>
    </div>

    <script type="module">
        import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

        let faceLandmarker;
        let video, dot, calibrationDiv;
        let isTracking = false;
        
        // Kalman Filter для сглаживания
        class KalmanFilter {
            constructor(R = 1, Q = 3) {
                this.R = R; // Шум измерений
                this.Q = Q; // Шум процесса
                this.A = 1;
                this.B = 0;
                this.C = 1;
                this.cov = NaN;
                this.x = NaN;
            }

            filter(z, u = 0) {
                if (isNaN(this.x)) {
                    this.x = (1 / this.C) * z;
                    this.cov = (1 / this.C) * this.Q * (1 / this.C);
                } else {
                    const predX = (this.A * this.x) + (this.B * u);
                    const predCov = ((this.A * this.cov) * this.A) + this.Q;
                    const K = predCov * this.C * (1 / ((this.C * predCov * this.C) + this.R));
                    this.x = predX + K * (z - (this.C * predX));
                    this.cov = predCov - (K * this.C * predCov);
                }
                return this.x;
            }
        }

        const kalmanX = new KalmanFilter(0.5, 2);
        const kalmanY = new KalmanFilter(0.5, 2);

        // Калибровочные данные
        const calibrationPoints = [];
        const calibrationData = [];
        let currentCalPoint = 0;

        document.getElementById('start-btn').onclick = async () => {
            video = document.getElementById('video');
            dot = document.getElementById('dot');
            calibrationDiv = document.getElementById('calibration');

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1920 },
                        height: { ideal: 1080 },
                        facingMode: 'user'
                    }
                });

                video.srcObject = stream;
                await video.play();

                const vision = await FilesetResolver.forVisionTasks(
                    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
                );

                faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                        delegate: 'GPU'
                    },
                    runningMode: 'VIDEO',
                    numFaces: 1,
                    minFaceDetectionConfidence: 0.7,
                    minFacePresenceConfidence: 0.7,
                    minTrackingConfidence: 0.7,
                    outputFaceBlendshapes: false,
                    outputFacialTransformationMatrixes: false
                });

                startCalibration();

            } catch (error) {
                alert('Ошибка доступа к камере: ' + error.message);
            }
        };

        function startCalibration() {
            const w = window.innerWidth;
            const h = window.innerHeight;
            const margin = 100;

            // 9-точечная калибровка для максимальной точности
            calibrationPoints.push(
                {x: w/2, y: h/2},           // Центр
                {x: margin, y: margin},      // Верхний левый
                {x: w/2, y: margin},         // Верхний центр
                {x: w-margin, y: margin},    // Верхний правый
                {x: w-margin, y: h/2},       // Правый центр
                {x: w-margin, y: h-margin},  // Нижний правый
                {x: w/2, y: h-margin},       // Нижний центр
                {x: margin, y: h-margin},    // Нижний левый
                {x: margin, y: h/2}          // Левый центр
            );

            showNextCalibrationPoint();
        }

        function showNextCalibrationPoint() {
            if (currentCalPoint >= calibrationPoints.length) {
                finishCalibration();
                return;
            }

            const point = calibrationPoints[currentCalPoint];
            const calPoint = document.createElement('div');
            calPoint.className = 'cal-point active';
            calPoint.style.left = point.x + 'px';
            calPoint.style.top = point.y + 'px';
            calibrationDiv.appendChild(calPoint);

            setTimeout(() => {
                collectCalibrationData(point, calPoint);
            }, 1000);
        }

        async function collectCalibrationData(point, calPoint) {
            const samples = [];
            const startTime = performance.now();
            
            while (performance.now() - startTime < 1500) {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    const results = faceLandmarker.detectForVideo(video, performance.now());
                    
                    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                        const eyeData = getEyeData(results.faceLandmarks[0]);
                        if (eyeData) {
                            samples.push({
                                screen: point,
                                eye: eyeData
                            });
                        }
                    }
                }
                await new Promise(resolve => setTimeout(resolve, 16));
            }

            if (samples.length > 10) {
                calibrationData.push(...samples);
            }

            calPoint.remove();
            currentCalPoint++;
            showNextCalibrationPoint();
        }

        function getEyeData(landmarks) {
            // Используем ирис ландмарки (468-477) для максимальной точности
            const leftIrisCenter = landmarks[468];
            const rightIrisCenter = landmarks[473];
            
            // Границы глаз
            const leftEyeLeft = landmarks[33];
            const leftEyeRight = landmarks[133];
            const leftEyeTop = landmarks[159];
            const leftEyeBottom = landmarks[145];
            
            const rightEyeLeft = landmarks[362];
            const rightEyeRight = landmarks[263];
            const rightEyeTop = landmarks[386];
            const rightEyeBottom = landmarks[374];

            // Нормализованные координаты ириса относительно глаза
            const leftGazeX = (leftIrisCenter.x - leftEyeLeft.x) / (leftEyeRight.x - leftEyeLeft.x);
            const leftGazeY = (leftIrisCenter.y - leftEyeTop.y) / (leftEyeBottom.y - leftEyeTop.y);
            
            const rightGazeX = (rightIrisCenter.x - rightEyeLeft.x) / (rightEyeRight.x - rightEyeLeft.x);
            const rightGazeY = (rightIrisCenter.y - rightEyeTop.y) / (rightEyeBottom.y - rightEyeTop.y);

            return {
                gazeX: (leftGazeX + rightGazeX) / 2,
                gazeY: (leftGazeY + rightGazeY) / 2,
                headX: (landmarks[1].x + landmarks[4].x) / 2,
                headY: (landmarks[1].y + landmarks[4].y) / 2
            };
        }

        function finishCalibration() {
            calibrationDiv.classList.add('hidden');
            isTracking = true;
            detectAndTrack();
        }

        function detectAndTrack() {
            if (!isTracking) return;

            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                const results = faceLandmarker.detectForVideo(video, performance.now());

                if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                    const eyeData = getEyeData(results.faceLandmarks[0]);
                    
                    if (eyeData && calibrationData.length > 0) {
                        const estimated = estimateGaze(eyeData);
                        
                        // Применяем Kalman Filter
                        const smoothX = kalmanX.filter(estimated.x);
                        const smoothY = kalmanY.filter(estimated.y);

                        dot.style.left = smoothX + 'px';
                        dot.style.top = smoothY + 'px';
                    }
                }
            }

            requestAnimationFrame(detectAndTrack);
        }

        function estimateGaze(currentEye) {
            // Находим ближайшие калибровочные точки методом k-NN (k=5)
            const distances = calibrationData.map(sample => ({
                sample,
                dist: Math.hypot(
                    currentEye.gazeX - sample.eye.gazeX,
                    currentEye.gazeY - sample.eye.gazeY
                )
            }));

            distances.sort((a, b) => a.dist - b.dist);
            const nearest = distances.slice(0, 5);

            // Взвешенное среднее по расстояниям
            let totalWeight = 0;
            let weightedX = 0;
            let weightedY = 0;

            nearest.forEach(({sample, dist}) => {
                const weight = 1 / (dist + 0.001);
                totalWeight += weight;
                weightedX += sample.screen.x * weight;
                weightedY += sample.screen.y * weight;
            });

            return {
                x: weightedX / totalWeight,
                y: weightedY / totalWeight
            };
        }

        // Пересчет при изменении размера окна
        window.addEventListener('resize', () => {
            if (isTracking) {
                alert('Для корректной работы не меняйте размер окна');
            }
        });
    </script>
</body>
</html>
