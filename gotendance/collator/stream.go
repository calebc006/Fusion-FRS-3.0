package collator

import (
	"bufio"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"
)


type Detection struct {
	Bbox []float32 `json:"bbox"`
	Label string `json:"label"`
	Score float32 `json:"score"`
} 


type Result struct {
	Data []Detection `json:"data"`
}


type StreamSrc struct {
	UpdateInterval time.Duration
	Url string
	StopChan chan struct{}
}


type StreamsList struct {
	mu sync.Mutex
	Items []StreamSrc
}


func NewStreamsList() *StreamsList {
	return &StreamsList{Items: make([]StreamSrc, 0)}
}


func (streamsList *StreamsList) AddStreamSrc(url string, updateInterval time.Duration) (chan struct{}) {
	streamsList.mu.Lock()
	defer streamsList.mu.Unlock()

	for _, streamSrc := range streamsList.Items {
		if streamSrc.Url == url {
			return nil 
		}
	}

	stopChan := make(chan struct{})
	newStreamSrc := StreamSrc{
		UpdateInterval: updateInterval,
		Url: url,
		StopChan: stopChan,
	}
	streamsList.Items = append(streamsList.Items, newStreamSrc)
	return stopChan
}


func (streamsList *StreamsList) RemoveStreamSrc(url string) {
	streamsList.mu.Lock()
	defer streamsList.mu.Unlock()

	for i, streamSrc := range streamsList.Items {
		if streamSrc.Url == url {
			streamsList.Items = append(streamsList.Items[:i], streamsList.Items[i+1:]...)
			close(streamSrc.StopChan)
			return
		}
 	}
}


func (streamsList *StreamsList) FetchList() []StreamSrc {
	streamsList.mu.Lock()
	defer streamsList.mu.Unlock()

	return streamsList.Items
}


func handleResult(result []Detection, store *Store) {
	for _, detection := range result {
		store.Check(detection.Label)
	}
}


func Stream(store *Store, stopChan chan struct{}, resultsUrl string, updateInterval time.Duration) {
	var (
		latestData []Detection
		mu sync.Mutex
	)

	updateStore := func() {
		ticker := time.NewTicker(updateInterval * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <- stopChan:
				log.Printf("Ending Handler Goroutine")
				return
			case <-ticker.C:
				mu.Lock()
				handleResult(latestData, store)
				mu.Unlock()
			}
		}
	}

	resp, err := http.Get(resultsUrl)
	if err != nil {
		log.Printf("Failed to make request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Received non-OK status; %s", resp.Status)
	}

	reader := bufio.NewReader(resp.Body)

	log.Printf("Receiving results from %s at intervals %v", resultsUrl, time.Duration(updateInterval * time.Millisecond))

	go updateStore()

	for {
		select {
		case <-stopChan:
			log.Printf("Ending stream Goroutine")
			return
		default:
			line, err := reader.ReadBytes('\n')
			if err != nil {
				log.Printf("Stream ended or error occured: %v", err)
				return
			}

			var result Result
			err = json.Unmarshal(line, &result) 
			if err != nil {
				log.Printf("Error unmarshaling JSON: %v", err)
				continue
			}

			mu.Lock()
			latestData = result.Data
			mu.Unlock()
		}
	}
}
