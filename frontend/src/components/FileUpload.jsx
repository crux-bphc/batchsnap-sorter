import { Upload, Button, Icon } from "antd";
import React from "react";

class FileUpload extends React.Component {
  state = {
    fileList: []
  };

  handleUpload = () => {
    const { fileList } = this.state;

    this.props.onUpload(fileList[0]);
  };

  render() {
    const { fileList } = this.state;
    const props = {
      onRemove: file => {
        this.setState(state => {
          const index = state.fileList.indexOf(file);
          const newFileList = state.fileList.slice();
          newFileList.splice(index, 1);
          return {
            fileList: newFileList
          };
        });
      },

      accept: ".jpg, .jpeg, .png",
      multiple: false,

      beforeUpload: file => {
        this.setState(state => ({
          fileList: [...state.fileList, file]
        }));
        return false;
      },

      fileList
    };

    return (
      <div>
        <h3>Upload an Image! It should not be a group picture.</h3>
        <Upload {...props}>
          <Button disabled={this.state.fileList.length >= 1 ? true : false}>
            <Icon type="upload" />
            Select File
          </Button>
        </Upload>
        <Button
          type="primary"
          onClick={this.handleUpload}
          hidden={fileList.length === 0}
          style={{ marginTop: 16 }}
        >
          Start Upload
        </Button>
      </div>
    );
  }
}

export default FileUpload;
