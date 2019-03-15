import React from "react";
import Webcam from "react-webcam";
import { Button } from "antd";

class Camera extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isCaptured: false,
      imgSrc: null,
      uploading: false
    };
  }

  _b64toBlob(b64Data, contentType, sliceSize) {
    contentType = contentType || "";
    sliceSize = sliceSize || 512;

    var byteCharacters = atob(b64Data);
    var byteArrays = [];

    for (var offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      var slice = byteCharacters.slice(offset, offset + sliceSize);

      var byteNumbers = new Array(slice.length);
      for (var i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }

      var byteArray = new Uint8Array(byteNumbers);

      byteArrays.push(byteArray);
    }

    var blob = new Blob(byteArrays, { type: contentType });
    return blob;
  }

  capture = () => {
    let imgSrc = this.webcam.getScreenshot();
    this.setState({ isCaptured: true, imgSrc: imgSrc });
  };

  upload = () => {
    // converting base64 to blob
    var block = this.state.imgSrc.split(";");
    var contentType = block[0].split(":")[1];
    var realData = block[1].split(",")[1];

    var blob = this._b64toBlob(realData, contentType);
    this.props.onUpload(blob);
  };

  reset = () => {
    this.setState({ isCaptured: false, imgSrc: null });
  };

  render() {
    if (this.state.isCaptured)
      var content = (
        <div>
          <img
            style={{
              marginBottom: "60px",
              marginTop: "60px"
            }}
            src={this.state.imgSrc}
            alt=""
          />
          <br />
          <Button
            style={{ marginTop: "5px", marginRight: "5px" }}
            onClick={this.upload}
          >
            Upload
          </Button>
          <Button onClick={this.state.isCaptured ? this.reset : this.capture}>
            {this.state.isCaptured ? "Try again" : "Capture"}
          </Button>
        </div>
      );
    else
      content = (
        <div>
          <Webcam
            height={500}
            width={500}
            audio={false}
            ref={node => {
              this.webcam = node;
            }}
            screenshotFormat="image/jpeg"
            videoConstraints={{
              facingMode: "user"
            }}
          />
          <Button onClick={this.state.isCaptured ? this.reset : this.capture}>
            {this.state.isCaptured ? "try again" : "capture"}
          </Button>
        </div>
      );
    return (
      <div
        style={{
          marginLeft: "33%",
          marginRight: "33%"
        }}
      >
        {content}
      </div>
    );
  }
}

export default Camera;
