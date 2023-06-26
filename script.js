const video = document.getElementById('video');

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(startWebcam);

function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            detectFaces();
        })
        .catch((error) => {
            console.error('Error accessing webcam:', error);
        });
}

async function loadLabeledImages() {
    const labels = ["Title", "Titus"];
    return Promise.all(
        labels.map(async (label) => {
            const descriptions = [];
            for (let i = 1; i <= 4; i++) {
                const img = await faceapi.fetchImage(
                    `labeled_images/${label}/${i}.jpg`
                );
                const detections = await faceapi
                    .detectSingleFace(img)
                    .withFaceLandmarks()
                    .withFaceDescriptor();
                descriptions.push(detections.descriptor);
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    );
}


function detectFaces() {
    video.addEventListener("play", async () => {
        const canvas = faceapi.createCanvasFromMedia(video);
        document.body.append(canvas);

        const displaySize = { width: video.width, height: video.height };
        faceapi.matchDimensions(canvas, displaySize);

        const labeledFaceDescriptors = await loadLabeledImages();
        const faceMatcher = new faceapi.FaceMatcher(
            labeledFaceDescriptors,
            0.6
        );

        setInterval(async () => {
            const detections = await faceapi
                .detectAllFaces(video)
                .withFaceLandmarks()
                .withFaceDescriptors();

            const resizedDetections = faceapi.resizeResults(
                detections,
                displaySize
            );

            canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
                resizedDetections.forEach((detection) => {
                    const box = detection.detection.box;
                    const label = faceMatcher.findBestMatch(detection.descriptor).toString();
                    const confidence = Math.round(
                        faceMatcher.findBestMatch(detection.descriptor).distance * 100
                    );

                    const drawBox = new faceapi.draw.DrawBox(box, {
                        label: `${faceMatcher.findBestMatch(detection.descriptor).toString()} (${Math.round(faceMatcher.findBestMatch(detection.descriptor)._distance * 100)}%)`
                    });
                    drawBox.draw(canvas);
                });
        }, 100);
    });
}

