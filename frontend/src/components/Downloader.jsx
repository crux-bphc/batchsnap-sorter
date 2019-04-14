import React, { Component } from "react";
import axios from "axios";
import { Modal, message, Progress } from "antd";
import JSZip from "jszip";
import { saveAs } from "file-saver";

class Downloader extends Component {
  constructor() {
    super();
    this.state = {
      progress: 0
    };
  }

  componentDidMount() {
    this.downloadImages();
  }

  downloadImages = async () => {
    const { links } = this.props;
    let count = 0;
    let zip = JSZip();

    for (let l of links) {
      let res = null;
      try {
        res = await axios.get(l, {
          responseType: "blob"
        });
      } catch (e) {
        message.error("Could not download Images");
        break;
      }

      const imageName = l.split("/")[2];
      zip.file(imageName, res.data, {
        binary: true
      });

      ++count;
      this.setState({ progress: Math.floor((count / links.length) * 100) });
    }

    if (count === links.length)
      try {
        const content = await zip.generateAsync({
          type: "blob"
        });
        saveAs(content, `batchsnap-pics-${new Date()}.zip`);
      } catch (e) {
        message.error("Error while zipping!");
      }
    this.close();
  };

  close = () => {
    this.props.onCompleteOrCancel();
  };

  render() {
    return (
      <Modal
        visible
        title="Downloading Images.."
        onCancel={this.close}
        footer={null}
        style={{
          textAlign: "center"
        }}
      >
        <h3>Please wait while the images are being downloaded.</h3>
        <Progress type="circle" percent={this.state.progress} />
      </Modal>
    );
  }
}

export default Downloader;
