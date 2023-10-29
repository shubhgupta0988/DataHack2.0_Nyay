
// // import Data from '../../public/assets/extracted_data_with_gender.csv';
// // import Papa from 'papaparse';
// // import {useEffect, useState} from 'react';
// // import {Bar} from 'react-chartjs-2';
// // import {
// //   Chart as ChartJS,
// //   CategoryScale,
// //   LinearScale,
// //   BarElement,
// //   Title,
// //   Tooltip,
// //   Legend,
// // } from 'chart.js';
 
// // ChartJS.register(
// //   CategoryScale,
// //   LinearScale,
// //   BarElement,
// //   Title,
// //   Tooltip,
// //   Legend,
// // )
 
// // function Dashboard() {
// //   const [chartData, setChartData] = useState({
// //     datasets: []
// //   });
// //   const [chartOptions, setChartOptions] = useState({})
 
// //   useEffect(() => {
// //     Papa.parse(Data, {
// //       download: true,
// //       header: true,
// //       dynamicTyping: true,
// //       delimiter: "",
// //       complete: ((result) => {
// //         console.log(result)
// //         // setChartData({
// //         //   labels: result.data.map((item, index) => [item['name']]).filter( String ),
// //         //   datasets: [
// //         //     {
// //         //       label: "OSCAR WINNER",
// //         //       data: result.data.map((item, index) => [item[' "Age"']]).filter( Number ),
// //         //       borderColor: "black",
// //         //       backgroundColor: "red"
// //         //     }
// //         //   ]
// //         // });
// //         // setChartOptions({
// //         //   responsive: true,
// //         //   plugins: {
// //         //     legend: {
// //         //       position: 'top'
// //         //     },
// //         //     title: {
// //         //       display: true,
// //         //       text: "ALL OSCAR WINNERS SINCE 1928"
// //         //     }
// //     //       }
// //     //     })
// //     //   })
// //     })
// //   }, []);

// //   return (
// //     <div>
// //       <h1>Dashboard</h1>
// //       {/* {
// //         chartData.datasets.length > 0 ? (
// //           <div>
// //             <Bar options={chartOptions} data={chartData}/>
// //             </div>
// //         ) : (
// //           <div>
// //             Loading...
// //             </div>
// //         )
// //       } */}
// //     </div>
// //   )
// // })
// // }
 
// // export default Dashboard



// import Data from '../../public/assets/output.csv';
// import React, { useEffect, useState } from 'react';
// import Papa from 'papaparse';
// import { Bar } from 'react-chartjs-2';
// import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// function Dashboard() {
//   const [chartData, setChartData] = useState({
//     labels: [],
//     datasets: [
//       {
//         label: 'OSCAR WINNER',
//         data: [],
//         borderColor: 'black',
//         backgroundColor: 'red',
//       },
//     ],
//   });

//   const [chartOptions, setChartOptions] = useState({
//     responsive: true,
//     plugins: {
//       legend: {
//         position: 'top',
//       },
//       title: {
//         display: true,
//         text: 'ALL OSCAR WINNERS SINCE 1928',
//       },
//     },
//   });

//   useEffect(() => {
//     Papa.parse(Data, {
//       download: true,
//       header: true,
//       dynamicTyping: true,
//       complete: (result) => {
//         const filteredData = result.data.filter((item) => item['name'] && !isNaN(item['Age']));

//         setChartData({
//           labels: filteredData.map((item) => item['name']),
//           datasets: [
//             {
//               label: 'OSCAR WINNER',
//               data: filteredData.map((item) => item['Age']),
//               borderColor: 'black',
//               backgroundColor: 'red',
//             },
//           ],
//         });
//       },
//     });
//   }, []);

//   return (
//     <div>
//       <h1>Dashboard</h1>
//       {chartData.labels.length > 0 ? (
//         <div>
//           <Bar options={chartOptions} data={chartData} />
//         </div>
//       ) : (
//         <div>Loading...</div>
//       )}
//     </div>
//   );
// }

// export default Dashboard;
