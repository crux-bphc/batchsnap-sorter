import React, { Component } from "react";
import { Card, Divider, Button, message, Icon } from "antd";
import FileUpload from "./components/FileUpload";
import Camera from "./components/Camera";
import Downloader from "./components/Downloader";
import axios from "axios";
import logo from "./assets/logo.png";
class App extends Component {
  state = {
    camera: false,
    upload: false,
    gotResponse: false,
    links: []
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
        console.log(response.links);
        this.setState({ links: response.links || [], gotResponse: true });
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
      links: [],
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
                  {this.state.links.length !== 0 && (
                    <div>
                      <Downloader
                        onCompleteOrCancel={this.reset}
                        links={this.state.links}
                      />
                    </div>
                  )}

                  {!this.state.links.length && (
                    <div>
                      <h2>Sorry, we couldn't find any pictures of you.</h2>
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
                Try Again.
              </Button>
            </Card>
          </div>
        )}

        <footer
          style={{
            marginTop: "5%"
          }}
        >
          <span>
            <h4
              style={{
                fontFamily: "helvetica"
              }}
            >
              Powered by CRUX
            </h4>
            <img
              style={{
                height: "50px",
                width: "50px"
              }}
              src={logo}
              alt="~~~~~~"
            />
          </span>
        </footer>
      </div>
    );
  }
}

export default App;
