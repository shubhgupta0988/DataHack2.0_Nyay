import React from "react";
// import { Link } from 'react-router-dom'
import styled from "styled-components";
import GavelIcon from "@mui/icons-material/Gavel";
import BusinessCenterIcon from "@mui/icons-material/BusinessCenter";
import StarIcon from "@mui/icons-material/Star";

const LawyerInfo = styled.div`
  width: 44rem;
  height: 12rem;
  background-color: white;
  border-radius: 0 0 10px 10px;
  border-top: 6px solid #0d265c;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 1rem;
`;

const LawyerDesc = styled.div`
  width: 95%;
  height: 50%;
  display: flex;
`;
const LawyerImgContainer = styled.div`
  flex: 1;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
`;
const LawyerImg = styled.img`
  width: 90%;
  height: 85%;

  border-radius: 100px;
`;
const LawyerNameBio = styled.div`
  flex: 6;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 5%;
  color: #0d265c;
  font-size: 14px;
  .name {
    font-weight: 700;
    font-size: 25px;
  }
`;
const LawyerStatsContainer = styled.div`
  width: 95%;
  height: 30%;
  display: flex;
  align-items: center;
`;
const LawyerType = styled.div`
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 14px;
  margin-right: 1rem;
`;
const LawyerExp = styled.div`
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 14px;
  margin-right: 1rem;
`;
const LawyerRating = styled.div`
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 14px;
  margin-right: 0.4rem;
`;
const Container = styled.div`
  margin:0.7rem 0 ;
  width: 44rem;
  height: 12rem;
  background-color: white;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items:center;
  .
`;
const Title = styled.div`
  width: 90%;
  font-weight: 600;
  font-size: 1.3rem;
  margin: 0.7rem;
`;
const SessionInfo = styled.div`
  height: 60%;
  width: 90%;
  border: 1px solid grey;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: space-around;
`;
const SessionDesc = styled.div`
  width: 70%;
  height: 70%;
  display: flex;
  flex-direction: column;
  border-right: 1px solid grey;
  justify-content: space-between;
  .title {
    font-weight: 600;
    font-size: 1.3rem;
  }
  .desc {
    font-size: 13px;
  }
  .price {
    width: 90%;
    text-align: right;
    font-weight: 800;
    font-size: 1.7rem;
  }
`;
const BookNow = styled.button`
  width: 20%;
  height: 40%;
  color: white;
  font-weight: 700;
  background-color: #0d265c;
  border-radius: 20px;
  border: none;
`;
const ExpertiseLangContainer = styled.div`
  margin: 0.7rem 0;
  width: 44rem;
  height: 6rem;
  background-color: white;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;
const ExpertiseLang = styled.div`
  width: 90%;
  height: 35%;
  display: flex;
  align-items: center;
  font-size: 13px;
  gap: 1rem;
  div {
    padding: 2%;
    height: 90%;
    font-weight: 500;
    font-size: 0.8rem;
    border-radius: 20px;
    border: 1px solid #0d265c;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .sort {
    background-color: #e5f1fcff;
  }
`;
const EducationContainer = styled.div`
  margin: 0.7rem 0;
  width: 44rem;
  height: auto;
  background-color: white;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const EduDesc = styled.div`
  width: 90%;
  height: 4.5rem;
  display: flex;
  margin-bottom: 2%;
`;
const EduImg = styled.img`
  width: 12%;
  height: 100%;
  background-color: yellow;
`;
const Education = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  margin-left: 2%;
  color: var(--primary-color);
  h2 {
    font-weight: 600;
  }
  div {
    font-size: 12px;
    margin-top: 1%;
    font-weight: 500;
  }
`;
const WorkExpContainer = styled.div`
  margin: 0.7rem 0;
  width: 44rem;
  height: 12rem;
  background-color: white;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;
const Wrapper = styled.div`
  width: 90%;
  display: flex;
  flex-direction: column;
  margin-bottom: 2%;
`;

const LawyerProfile = (props) => {
  console.log(props.lawyerData);
  return (
    <div>
      <LawyerInfo>
        <LawyerDesc>
          <LawyerImgContainer>
            <LawyerImg />
          </LawyerImgContainer>
          <LawyerNameBio>
            <div className="name">{props.lawyerData.Name}</div>
          </LawyerNameBio>
        </LawyerDesc>
        <LawyerStatsContainer>
          <LawyerType>
            <GavelIcon sx={{ margin: "0.2rem" }} />
            {props.lawyerData.Category}
          </LawyerType>

          <LawyerExp>
            <BusinessCenterIcon sx={{ margin: "0.2rem" }} />
            {props.lawyerData.Exp} Years
          </LawyerExp>

          <LawyerRating>
            <StarIcon sx={{ margin: "0.2rem", color: "yellow" }} />
            {props.lawyerData.Ratings}
          </LawyerRating>
        </LawyerStatsContainer>
      </LawyerInfo>

      <Container>
        <Title>Sessions</Title>
        <SessionInfo>
          <SessionDesc>
            <div className="title">Legal Consultation</div>
            <div className="desc">1:1 Video Call</div>
            <div className="price">$200</div>
          </SessionDesc>
          <BookNow>
            <Link to="/user/book">Book Now</Link>
          </BookNow>
        </SessionInfo>
      </Container>

      <ExpertiseLangContainer>
        <Title>Expertise</Title>
        <ExpertiseLang>{props.lawyerData.Category}</ExpertiseLang>
      </ExpertiseLangContainer>

      <ExpertiseLangContainer>
        <Title>Fluent In</Title>
        <ExpertiseLang>{props.lawyerData.Language}</ExpertiseLang>
      </ExpertiseLangContainer>

      <EducationContainer>
        <Title>Education</Title>
        <Wrapper>
          <EduDesc>
            <EduImg></EduImg>
            <Education>
              <h2>St. Xaviers</h2>
              <div> MBA</div>
              <div>2009-2010</div>
            </Education>
          </EduDesc>

          <EduDesc>
            <EduImg></EduImg>
            <Education>
              <h2>St. Xaviers</h2>
              <div> MBA</div>
              <div>2009-2010</div>
            </Education>
          </EduDesc>

          <EduDesc>
            <EduImg></EduImg>
            <Education>
              <h2>St. Xaviers</h2>
              <div> MBA</div>
              <div>2009-2010</div>
            </Education>
          </EduDesc>
        </Wrapper>
      </EducationContainer>

      <WorkExpContainer></WorkExpContainer>
    </div>
  );
};

export default LawyerProfile;
