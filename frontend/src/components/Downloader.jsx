import React, { Component } from "react";
import axios from "axios";
import { Modal, message, Progress } from "antd";
import JSZip from "jszip";
import { saveAs } from "file-saver";

class Downloader extends Component {
  constructor() {
    super();
    this.state = {
      progress: 0,
      zipping: false,
      cancelDownload: false
    };
  }

  componentDidMount() {
    this.downloadImages();
  }

  downloadImages = async () => {
    const { links } = this.props;
    let count = 0;
    let currentZipSize = 0;
    let currentZip = JSZip();

    for (let l of links) {
      if (this.state.cancelDownload) {
        this.close();
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
      const imageName = l.split(/^\/images\//)[1];
      currentZip.file(imageName, res.data, {
        binary: true
      });

      if (currentZipSize > 475 * 1024 * 1024 || count === links.length) {
        const hide = message.loading("Zipping downloaded images!", 1000);
        currentZip
          .generateAsync({
            type: "blob"
          })
          .then(content => {
            saveAs(content, `batchsnap-pics-${new Date().getTime()}.zip`);
          })
          .catch(_err => {
            message.error("Error while zipping!");
          })
          .finally(() => {
            hide();
          });

        currentZip = JSZip(); // new zip
        currentZipSize = 0;
      }

      this.setState({ progress: Math.floor((count / links.length) * 100) });
    }
    message.success("Download Complete!");
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
        onCancel={() => {
          if (!this.state.zipping) this.setState({ cancelDownload: true });
        }}
        footer={null}
        style={{
          textAlign: "center"
        }}
      >
        <div>
          {this.state.progress !== 100 ? (
            <h3>Please wait while the images are being downloaded.</h3>
          ) : (
            <h3>Download Successful!</h3>
          )}
          <Progress type="circle" percent={this.state.progress} />
        </div>
      </Modal>
    );
  }
}

export default Downloader;
