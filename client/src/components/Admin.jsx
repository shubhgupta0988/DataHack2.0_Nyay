import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import styled from "styled-components";
const Hero = styled.div`
  width: 100%;
  height: 68vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #e7f3f8;
`;
const HeroContent = styled.div`
  flex: 4;
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 0 0;
  span {
    width: 60%;
    margin-left: 18%;
  }
  .herotitle {
    font-weight: 900;
    font-size: 5rem;
    margin-top: 5%;
    background: linear-gradient(
      to right,
      #0d265c 10%,
      #0b98da 30%,
      #0b98da 70%,
      #0d265c 80%
    );
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    text-fill-color: transparent;
    background-size: 300% auto;
    animation: textShine 5s ease-in-out infinite alternate;
    @keyframes textShine {
      0% {
        background-position: 0% 50%;
      }
      100% {
        background-position: 100% 50%;
      }
    }
  }
  .tagline {
    font-weight: 700;
    font-size: 2.3rem;

    background: linear-gradient(
      to right,
      #0d265c 10%,
      #0b98da 30%,
      #0b98da 70%,
      #0d265c 80%
    );
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    text-fill-color: transparent;
    background-size: 300% auto;
    animation: textShine 5s ease-in-out infinite alternate;
    @keyframes textShine {
      0% {
        background-position: 0% 50%;
      }
      100% {
        background-position: 100% 50%;
      }
    }
  }
  .description {
    font-weight: 500;
    font-size: 1.2rem;
    margin-top: 5%;
    color: grey;
  }
`;
const HeroImgContainer = styled.div`
  flex: 3;
  height: 100%;
  z-index: 1;
  position: relative;
  display: flex;
  justify-content: center;
  overflow: hidden;
  border-radius: 0 0 0 50px;
`;
const HeroImg = styled.div`
  width: 80%;
  height: 80%;
  position: absolute;
  z-index: 4;
  display: flex;
  img {
    margin-top: 8%;
    margin-left: 1%;
    width: 110%;
    height: 100%;
    object-fit: cover;
  }
`;
const DragContainer = styled.div`
  width: 80%;
  height: auto;
  display: flex;
  align-items: center;
  flex-direction: column;
  margin: auto;
  margin-top: 2%;
  h1 {
    margin-bottom: 2%;
  }
`;
const fileTypes = ["CSV", "XLSX"];

const DocTrans = () => {
  const [file, setFile] = useState(null);
  const handleChange = (file) => {
    setFile(file);
    const formData = new FormData();
    formData.append("file", file);
    const sendDoc = async () => {
      const response = await fetch("http://127.0.0.1:5000/change", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        console.log("error");
      } else {
        const data = await response.json();
        console.log(data);
      }
    };
    sendDoc();
  };

  return (
    <div>
      <Hero>
        <HeroContent>
          <span className="herotitle">ADMIN</span>
          <span className="tagline">
            Monitoring & Adding Data At Your Fingertips !
          </span>
          <span className="description">
            "Add new lawyer data and monitor ratings !"
          </span>
        </HeroContent>
        <HeroImgContainer>
          <div className="relative w-full max-w-lg">
            <div className="absolute top-5 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
            <div className="absolute top-4 right-0 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
            <div className="absolute top-60 left-0 w-72 h-72 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
            <div className="absolute top-40 left-60 w-72 h-72 bg-blue-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
          </div>
          <HeroImg>{/* <img src={findlawyerhero}></img> */}</HeroImg>
        </HeroImgContainer>
      </Hero>

      <DragContainer>
        <h1>Upload your file here</h1>
        <FileUploader
          handleChange={handleChange}
          name="file"
          types={fileTypes}
        />
      </DragContainer>
    </div>
  );
};

export default DocTrans;
