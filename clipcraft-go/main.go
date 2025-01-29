package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	// Serve static index.html on GET "/"
	r.GET("/", func(c *gin.Context) {
		c.File("static/index.html")
	})

	// Handle JSON request for video processing
	r.POST("/process", func(c *gin.Context) {
		var req VideoRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		finalPath, err := ProcessVideoRequest(req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"output_path": finalPath})
	})

	r.Run(":3000")
}
