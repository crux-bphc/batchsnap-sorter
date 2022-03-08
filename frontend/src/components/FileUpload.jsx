import { Upload, Button, Icon } from "antd";
import React from "react";
function FileUpload(){
  const [fileList,setFileList] = useState([]);
  const handleUpload = () => {
    props.onUpload(fileList[0]);
  }
    const props = {
      onRemove : file => {
          const index = fileList.indexOf(file);
          setFileList(fileList.slice().splice(index, 1));
          return {
            fileList: fileList
          };
        },

      accept: ".jpg, .jpeg, .png",
      multiple: false,

      beforeUpload: file => {
        setFileList([...fileList, file]);
        return false;
      },

      fileList
    };

    return (
      <div>
        <h3>Upload an Image! It should not be a group picture.</h3>
        <Upload {...props}>
          <Button disabled={fileList.length >= 1 ? true : false}>
            <Icon type="upload" />
            Select File
          </Button>
        </Upload>
        <Button
          type="primary"
          onClick={handleUpload}
          hidden={fileList.length === 0}
          style={{ marginTop: 16 }}
        >
          Start Upload
        </Button>
      </div>
    );
}

export default FileUpload;
