// src/axiosInterceptor.js
import axios from "axios";

const setupAxiosInterceptors = () => {
  // Add a request interceptor
  axios.interceptors.request.use(
    (config) => {
      const token = localStorage.getItem("token");
      if (token) {
        config.headers["x-auth-token"] = token; // Attach token to headers
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );
};

export default setupAxiosInterceptors;
