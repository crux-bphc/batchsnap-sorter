import React from "react";
import axios from "axios";
import { Modal, message, Progress, Icon } from "antd";
import JSZip from "jszip";
import { saveAs } from "file-saver";


function Downloader() {
const [progress,setProgress] = useState(0);
const [zipping,setZipping] = useState(false);
const [cancelDownload,setCancelDownload] = useState(false);

useEffect(()=>{
  downloadImages();
})
const downloadImages = async() =>{
  const { links } = props;
  let count = 0;
  let zips = [];
  let currentZipSize = 0;

  let currentZip = JSZip();

  for (let l of links) {
    if (cancelDownload) {
      close();
      return;
    }

    let res = null;
    try {
      res = await axios.get(l, {
        responseType: "blob"
      });
    } catch (e) {
      message.error("Could not download Images");
      break;
    }

    ++count;
    currentZipSize += res.data.size;

    if (currentZipSize > 499 * 1024 * 1024 || count % 200 === 0) {
      zips.push(currentZip);
      currentZip = JSZip(); // new zip
      currentZipSize = res.data.size;
    }

    const imageName = l.split(/^\/images\//)[1];
    currentZip.file(imageName, res.data, {
      binary: true
    });

    setProgress(Math.floor((count / links.length) * 100));
  }

  if (Object.keys(currentZip.files).length > 0) {
    zips.push(currentZip);
  }

  setZipping(true);

  for (let zip of zips) {
    try {
      let content = await zip.generateAsync({
        type: "blob"
      });
      saveAs(content, `batchsnap-pics-${new Date().getTime()}.zip`);
    } catch (err) {
      message.error("Error while zipping!");
      break;
    }
  }
  message.info("Zipping complete!");
  close();
}
const close =()=>{
  props.onCompleteOrCancel();
}
return (
  <Modal
    visible
    title="Downloading Images.."
    onCancel={() => {
      if (!zipping) setCancelDownload(true);
    }}
    footer={null}
    style={{
      textAlign: "center"
    }}
  >
    <div>
      {progress !== 100 ? (
        <h3>Please wait while the images are being downloaded.</h3>
      ) : (
        <h3>Download Successful!</h3>
      )}
      <Progress type="circle" percent={progress} />
      {zipping && (
        <>
          <h3>Zipping all your pictures..</h3>
          <Icon type="loading" />
        </>
      )}
    </div>
  </Modal>
);
}

export default Downloader;
