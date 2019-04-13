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

  downloadImages = () => {
    const { links } = this.props;
    let count = 0;
    let zip = JSZip();

    console.log(links);

    links.forEach(l => {
      axios
        .get(l, {
          responseType: "blob"
        })
        .then(res => {
          zip.file(`image${count}`, res.data, {
            binary: true
          });

          ++count;
          this.setState({ progress: (count / links.length) * 100 });

          if (count === links.length) {
            zip
              .generateAsync({
                type: "blob"
              })
              .then(content => {
                saveAs(content, new Date() + ".zip");
                this.close();
              })
              .catch(err => {
                message.error("Error while zipping!");
                this.close();
              });
          }
        })
        .catch(err => {
          message.error("Could not download Images");
          this.close();
        });
    });
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
