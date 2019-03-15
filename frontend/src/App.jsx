import React, { Component } from "react";
import { Card, Divider, Button, message, Icon } from "antd";
import FileUpload from "./components/FileUpload";
import Camera from "./components/Camera";
import axios from "axios";

class App extends Component {
  state = {
    camera: false,
    upload: false,
    gotResponse: false,
    link: ""
  };

  toggleCamera = () => {
    this.setState({ camera: !this.state.camera });
  };

  onUpload = file => {
    let formData = new FormData();
    formData.append("image", file);
    axios
      .post("/image/", formData)
      .then(res => {
        let response = res.data;
        this.setState({ link: response.link || "", gotResponse: true });
        message.success("Uploaded successfully");
      })
      .catch(err => {
        message.error("An error occurred!");
        this.reset();
      });
    this.setState({ upload: true });
  };

  reset = () => {
    this.setState({
      upload: false,
      link: "",
      gotResponse: false,
      camera: false
    });
  };

  render() {
    return (
      <div style={{ textAlign: "center", marginTop: "5%" }}>
        <h1>Welcome to BatchSnap Sorter!</h1>

        {!this.state.camera && !this.state.upload && (
          <div
            style={{
              marginLeft: "33%",
              marginTop: "2.5%",
              marginRight: "33%",
              textAlign: "center"
            }}
          >
            <Card>
              <FileUpload onUpload={this.onUpload} />
            </Card>
            <Divider> OR</Divider>
          </div>
        )}

        {!this.state.upload && (
          <Button onClick={this.toggleCamera}>
            {!this.state.camera ? "Click a picture" : "Close Camera"}
          </Button>
        )}

        {this.state.camera && !this.state.upload && (
          <Camera onUpload={this.onUpload} />
        )}

        {this.state.upload && (
          <div
            style={{
              marginLeft: "33%",
              marginTop: "2.5%",
              marginRight: "33%",
              textAlign: "center"
            }}
          >
            <Card>
              {!this.state.gotResponse && (
                <div>
                  <h2>Please wait while we fetch your pictures.. </h2>
                  <Icon type="loading" />
                </div>
              )}

              {this.state.gotResponse && (
                <div>
                  {this.state.link !== "" && (
                    <div>
                      <h2>Click the link to download</h2>
                      <a href={this.state.link}>Download</a>
                    </div>
                  )}

                  {this.state.link === "" && (
                    <div>
                      <h2>Sorry, we couldn't find any pictures of you</h2>
                    </div>
                  )}
                </div>
              )}

              <Button
                style={{
                  marginTop: "2.5%"
                }}
                onClick={this.reset}
              >
                Try Again with a new picture
              </Button>
            </Card>
          </div>
        )}
      </div>
    );
  }
}

export default App;
