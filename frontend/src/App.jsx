import React from "react";
import { Card, Button, message, Icon } from "antd";
import FileUpload from "./components/FileUpload";
import Downloader from "./components/Downloader";
import axios from "axios";
import logo from "./assets/logo.png";



function App(){
  const [upload,setUpload] = useState(false);
  const [gotResponse,setGotResponse] = useState(false);
  const [links,setLinks] = useState([]);
  const onUpload = e =>{
    let formData = new FormData();
    formData.append("image", file);
    axios
      .post("/image/", formData)
      .then(res => {
        let response = res.data;
        console.log(response.links);
        setLinks(response.links || []);
        setGotResponse(true);
        message.success("Uploaded successfully");
      })
      .catch(err => {
        message.error("An error occurred!");
        reset();
      });
    setUpload(true);
  }
  const reset = () => {
    setUpload(false);
    setLinks([]);
    setGotResponse(false);
  }
  return (
      <div style={{ textAlign: "center", marginTop: "5%" }}>
        <h1>Welcome to BatchSnap Sorter!</h1>

        {!upload && (
          <div
            style={{
              marginLeft: "33%",
              marginTop: "2.5%",
              marginRight: "33%",
              textAlign: "center"
            }}
          >
            <Card>
              <FileUpload onUpload={onUpload} />
            </Card>
          </div>
        )}

        {upload && (
          <div
            style={{
              marginLeft: "33%",
              marginTop: "2.5%",
              marginRight: "33%",
              textAlign: "center"
            }}
          >
            <Card>
              {!gotResponse && (
                <div>
                  <h2>Please wait while we fetch your pictures.. </h2>
                  <Icon type="loading" />
                </div>
              )}

              {gotResponse && (
                <div>
                  {links.length !== 0 && (
                    <div>
                      <Downloader
                        onCompleteOrCancel={reset}
                        links={links}
                      />
                    </div>
                  )}

                  {!links.length && (
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
                onClick={reset}
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

export default App;
