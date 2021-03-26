const fs = require("fs");
// const path = require("path");
const pixels = require("image-pixels");
// const yaml = require("js-yaml");
// const tf = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs-node");
const tfjsWasm = require("@tensorflow/tfjs-backend-wasm");
const faceLandmarksDetection = require("@tensorflow-models/face-landmarks-detection");

// tfjsWasm.setWasmPaths(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const calculateData = async () => {
    await tf.setBackend("wasm");
    console.log(new Date(), "tf.getBackend:", tf.getBackend());
    const model = await faceLandmarksDetection.load(faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);
    console.log(new Date(), "load model done");
    for (i = 0; i < 10; i++) {
        let fileBuffer = fs.readFileSync("./stream.jpg");
        let pixelStartTime = Date.now();
        let imageData = await pixels(fileBuffer);
        console.log(new Date(), `decode to Type ImageData  cost ${Date.now() - pixelStartTime} ms`);
        let startTime = Date.now();
        let faces = await model.estimateFaces({ input: imageData });
        console.log(new Date(), `google facemesh estimateFaces Done cost ${Date.now() - startTime} ms`);
        // console.log("faces:", faces[0]);
    }
    // let startTime = Date.now();
    // let faces = await model.estimateFaces({ input: imageData });
    // console.log(`estimateFaces1 Done cost ${Date.now() - startTime}`);
    // // console.log("faces:", faces[0]);
    // startTime = Date.now();
    // faces = await model.estimateFaces({ input: imageData });
    // console.log(`estimateFaces2 Done cost ${Date.now() - startTime}`);
    // // console.log("faces:", faces[0]);
    // startTime = Date.now();
    // faces = await model.estimateFaces({ input: imageData });
    // console.log(`estimateFaces2 Done cost ${Date.now() - startTime}`);
    // console.log("faces:", faces[0]);
};

calculateData();
