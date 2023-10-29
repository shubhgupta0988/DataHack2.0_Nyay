import React from "react";
import styled from "styled-components";
import StarIcon from "@mui/icons-material/Star";

const Card = styled.div`
  width: 70rem;
  height: 8rem;
  border-radius: 10px;
  background-color: white;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;

  margin-bottom: 1%;
  cursor: pointer;
  &:hover {
    ${"" /* border:1px solid #0B98DA; */}
    box-shadow: 0 0 4px  hsl(0, 0%, 81%);
  }
`;
const LawyerInfo = styled.div`
  width: 95%;
  height: 55%;
  display: flex;
`;
const LawyerImg = styled.div`
  width: 8%;
  height: 90%;
  border-radius: 100px;
  background-color: blue;
`;
const LawyerDesc = styled.div`
  flex: 5;
  display: flex;
  width: 70%;
  flex-direction: column;
  justify-content: center;
  gap: 10%;
  margin-left: 4%;
  .name {
    font-weight: 700;
    font-size: 20px;
    color: #0d265c;
  }
  .shortDesc {
    font-size: 15px;
  }
`;
const LaywerStars = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  .rating {
    font-weight: 700;
    margin-left: 2%;
  }
`;
const LawyerCategories = styled.div`
  display: flex;
  justify-content: center;
  gap: 2%;
  width: 95%;
  height: 30%;
  .catName {
    font-size: 12px;
    font-weight: 500;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #e5f1fcff;
    height: 100%;
    width: 50%;
    border-radius: 20px;
  }
`;
const LawyerListCard = (props) => {
  // Sample input
  console.log(props);
  const inputString = props.lawyerData.Category;
  const laws = inputString.split(" law");
  return (
    <div>
      <Card>
        <LawyerInfo>
          <LawyerImg />
          <LawyerDesc>
            <div className="name">{props.lawyerData.Name}</div>
            <div className="shortDesc">
              Experience :{props.lawyerData.Exp} years | Location :
              {props.lawyerData.Location}{" "}
            </div>
          </LawyerDesc>

          <LaywerStars>
            <StarIcon sx={{ color: "#fcb903ff" }} />
            <div className="rating">{props.lawyerData.Ratings}</div>
          </LaywerStars>
        </LawyerInfo>

        <LawyerCategories>
          {laws.map((law, index) => (
            <div className="catName">{law.trim()}</div>
            // <li key={index}>{law.trim()} </li>
          ))}

          {/* <div className="catName">{props.lawyerData.Category}</div> */}
        </LawyerCategories>
      </Card>
    </div>
  );
};

export default LawyerListCard;
