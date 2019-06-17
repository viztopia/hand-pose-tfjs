let model;
let imageToTrack;
let button;
let handX = [];
let handY = [];

const IMAGE_SIZE = 368;
const jointNumber = 21;

loadModel();

function preload() {
    // imageToTrack = createImg("images/hand.jpg");
    // imageToTrack.hide();
}

function setup() {
    createCanvas(600, 400);
    button = createButton('track hand');
    button.mousePressed(startTracking);
    imageToTrack = createCapture(VIDEO);
    imageToTrack.size(500, 400);
    imageToTrack.hide();

}

function draw() {
    background(255);
    image(imageToTrack, 0, 0);


    if (handX.length > 0 && handY.length > 0) {
        // console.log(handX.length);
        for (let i = 0; i < handX.length; i++) {
            noStroke();
            fill(232, 122, 19);
            ellipse(handX[i], handY[i], 10);
        }
    }

    if (handX.length == 21 && handY.length == 21) {
        stroke(255, 170, 32);
        strokeWeight(1);

        // console.log(handX.length);
        for (let i = 0; i < 4; i++) {
            line(handX[i], handY[i], handX[i + 1], handY[i + 1]);
        }

        line(handX[0], handY[0], handX[5], handY[5]);

        for (let i = 5; i < 8; i++) {
            line(handX[i], handY[i], handX[i + 1], handY[i + 1]);
        }

        line(handX[0], handY[0], handX[9], handY[9]);

        for (let i = 9; i < 12; i++) {
            line(handX[i], handY[i], handX[i + 1], handY[i + 1]);
        }

        line(handX[0], handY[0], handX[13], handY[13]);

        for (let i = 13; i < 16; i++) {
            line(handX[i], handY[i], handX[i + 1], handY[i + 1]);
        }

        line(handX[0], handY[0], handX[17], handY[17]);

        for (let i = 17; i < 20; i++) {
            line(handX[i], handY[i], handX[i + 1], handY[i + 1]);
        }
    }


}

async function loadModel() {
    model = await tf.loadLayersModel('model/model.json');
    model.summary();
}

function startTracking() {
    trackHand(imageToTrack);
}

async function trackHand(imageToTrack) {

    await tf.nextFrame();

    // console.log("------------instance of video? --------------")
    // console.log(imageToTrack.elt instanceof HTMLVideoElement);
    //
    // console.log("------------video ready? --------------")
    // console.log(imageToTrack.elt.readyState);
    if (typeof imageToTrack === 'object' && imageToTrack.elt instanceof HTMLVideoElement) {

        // console.log("=====================video!!!!!!!!!====================")
        const video = imageToTrack.elt;

        // Wait for the video to be ready
        if (video && video.readyState === 0) {
            await new Promise(resolve => {
                video.onloadeddata = () => resolve();
            });
        }

        await tf.nextFrame();
        const result = tf.tidy(() => {
            const imageResize = [IMAGE_SIZE, IMAGE_SIZE];
            const processedImg = imgToTensor(video, imageResize);
            // console.log(model);
            const predictions = model.predict(processedImg);
            // console.log(predictions)
            return predictions;
        });
        // console.log(result);

        const lastHeatMap = result[result.length - 1].squeeze();
        // console.log("lastHeatMap:");
        // console.log(lastHeatMap);

        for (let currnetJoint = 0; currnetJoint < jointNumber; currnetJoint++) {
            // console.log("heat map shape:");
            // console.log(lastHeatMap.shape);
            const joint1_HM_3d = lastHeatMap.slice([0, 0, 0 + currnetJoint], [46, 46, 1]).clone();
            // console.log("joint1_3d:");
            // console.log(joint1_HM_3d);

            const joint1_HM_2d = joint1_HM_3d.reshape([46, 46]);
            // console.log("joint1_2d:");
            // console.log(joint1_HM_2d);

            joint1_HM_2d.array().then((array) => {
                // console.log("joint heat map array");
                // console.log(array);
                let array_flat = array.flat();
                let indexOfMaxValue = array_flat.indexOf(Math.max(...array_flat));
                // console.log("max value & index:");
                // console.log(Math.max(...array_flat))
                // console.log(indexOfMaxValue);
                let cordY = Math.floor(indexOfMaxValue / 46);
                let cordX = indexOfMaxValue % 46;
                // console.log("X: " + cordX + " Y: " + cordY);
                let scaleUnitX = video.width / 46;
                let scaleUnitY = video.height / 46;
                handX[currnetJoint] = scaleUnitX * cordX;
                handY[currnetJoint] = scaleUnitY * cordY;
            })

            joint1_HM_3d.dispose();
            joint1_HM_2d.dispose();
        }

        // result.dispose();
        lastHeatMap.dispose();


        trackHand(imageToTrack);


    } else if (imageToTrack) {
        await tf.nextFrame();
        const result = tf.tidy(() => {
            const imageResize = [IMAGE_SIZE, IMAGE_SIZE];
            const processedImg = imgToTensor(imageToTrack.elt, imageResize);
            // console.log(model);
            const predictions = model.predict(processedImg);
            // console.log(predictions)
            return predictions;
        });
        // console.log(result);

        const lastHeatMap = result[result.length - 1].squeeze();
        // console.log("lastHeatMap:");
        // console.log(lastHeatMap);

        for (let currnetJoint = 0; currnetJoint < jointNumber; currnetJoint++) {
            console.log("heat map shape:");
            console.log(lastHeatMap.shape);
            let joint1_HM_3d = lastHeatMap.slice([0, 0, 0 + currnetJoint], [46, 46, 1]).clone();
            console.log("joint1_3d:");
            console.log(joint1_HM_3d);

            let joint1_HM_2d = joint1_HM_3d.reshape([46, 46]);
            console.log("joint1_2d:");
            console.log(joint1_HM_2d);

            joint1_HM_2d.array().then((array) => {
                console.log("joint heat map array");
                console.log(array);
                let array_flat = array.flat();
                let indexOfMaxValue = array_flat.indexOf(Math.max(...array_flat));
                console.log("max value & index:");
                console.log(Math.max(...array_flat))
                console.log(indexOfMaxValue);
                let cordY = Math.floor(indexOfMaxValue / 46);
                let cordX = indexOfMaxValue % 46;
                console.log("X: " + cordX + " Y: " + cordY);
                let scaleUnitX = imageToTrack.width / 46;
                let scaleUnitY = imageToTrack.height / 46;
                handX[currnetJoint] = scaleUnitX * cordX;
                handY[currnetJoint] = scaleUnitY * cordY;
            })
        }
    }
}

// Static Method: crop the image
const cropImage = (img) => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
};

function imgToTensor(input, size = null) {
    return tf.tidy(() => {
        let img = tf.browser.fromPixels(input);

        if (size) {
            img = tf.image.resizeBilinear(img, size);
        }
        // console.log(img)
        const croppedImage = cropImage(img);
        const batchedImage = croppedImage.expandDims(0);
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

function visualizeResult() {

}
