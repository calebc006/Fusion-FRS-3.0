package collator

import (
	"bufio"
	"encoding/json"
	"fmt"
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
	Health StreamHealth
}

type StreamHealth struct {
	mu              sync.RWMutex
	IsHealthy       bool
	LastError       string
	LastErrorTime   time.Time
	ConnectionCount int
	FailureCount    int
	LastPing        time.Time
}

func NewStreamHealth() StreamHealth {
	return StreamHealth{
		IsHealthy:    false,
		LastPing:     time.Now(),
		ConnectionCount: 0,
		FailureCount: 0,
	}
}

func (sh *StreamHealth) SetHealthy() {
	sh.mu.Lock()
	defer sh.mu.Unlock()
	sh.IsHealthy = true
	sh.LastPing = time.Now()
}

func (sh *StreamHealth) SetUnhealthy(err string) {
	sh.mu.Lock()
	defer sh.mu.Unlock()
	sh.IsHealthy = false
	sh.LastError = err
	sh.LastErrorTime = time.Now()
	sh.FailureCount++
}

func (sh *StreamHealth) GetStatus() (healthy bool, lastError string, failureCount int) {
	sh.mu.RLock()
	defer sh.mu.RUnlock()
	return sh.IsHealthy, sh.LastError, sh.FailureCount
}

func (sh *StreamHealth) IncrementConnectionCount() {
	sh.mu.Lock()
	defer sh.mu.Unlock()
	sh.ConnectionCount++
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
		Health: NewStreamHealth(),
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

func (streamsList *StreamsList) UpdateStreamHealth(url string, healthy bool, errMsg string) {
	streamsList.mu.Lock()
	defer streamsList.mu.Unlock()

	for i, streamSrc := range streamsList.Items {
		if streamSrc.Url == url {
			if healthy {
				streamsList.Items[i].Health.SetHealthy()
			} else {
				streamsList.Items[i].Health.SetUnhealthy(errMsg)
			}
			return
		}
	}
}


func handleResult(result []Detection, store *Store, filename string) {
	for _, detection := range result {
		store.Check(detection.Label)
	}
	store.JsonSave(filename)
}


func Stream(store *Store, streamsList *StreamsList, stopChan chan struct{}, resultsUrl string, updateInterval time.Duration, filename string) {
	var (
		detections_queue = make(map[string]Detection)
		mu sync.Mutex
		maxRetries = 5
		retryCount = 0
		baseBackoff = time.Second
	)

	updateStore := func() {
		ticker := time.NewTicker(updateInterval * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <- stopChan:
				log.Printf("Ending Handler Goroutine for %s", resultsUrl)
				return
			case <-ticker.C:
				mu.Lock()
				for _, det := range detections_queue {
					handleResult([]Detection{det}, store, filename)
				}
				detections_queue = make(map[string]Detection) // clear for next interval
				mu.Unlock()
			}
		}
	}

	go updateStore()

	// Retry loop with exponential backoff
	for {
		select {
		case <-stopChan:
			log.Printf("Ending stream Goroutine for %s", resultsUrl)
			return
		default:
			// Attempt to connect and stream
			err := streamWithRetry(store, streamsList, stopChan, resultsUrl, updateInterval, filename, &detections_queue, &mu)
			
			if err != nil {
				retryCount++
				streamsList.UpdateStreamHealth(resultsUrl, false, err.Error())
				if retryCount >= maxRetries {
					log.Printf("Stream %s failed after %d retries. Giving up.", resultsUrl, maxRetries)
					return
				}

				// Exponential backoff: 1s, 2s, 4s, 8s, etc.
				backoffDuration := baseBackoff * time.Duration(1<<uint(retryCount-1))
				log.Printf("Stream %s error: %v. Retrying in %v (attempt %d/%d)", resultsUrl, err, backoffDuration, retryCount, maxRetries)
				
				select {
				case <-stopChan:
					return
				case <-time.After(backoffDuration):
					continue
				}
			} else {
				// Connection successful, reset retry count
				retryCount = 0
			}
		}
	}
}

func streamWithRetry(store *Store, streamsList *StreamsList, stopChan chan struct{}, resultsUrl string, updateInterval time.Duration, filename string, detections_queue *map[string]Detection, mu *sync.Mutex) error {
	resp, err := http.Get(resultsUrl)
	if err != nil {
		return fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("received non-OK status: %s", resp.Status)
	}

	reader := bufio.NewReader(resp.Body)
	log.Printf("Receiving results from %s at intervals %v", resultsUrl, time.Duration(updateInterval*time.Millisecond))
	
	// Mark as healthy since connection was successful
	streamsList.UpdateStreamHealth(resultsUrl, true, "")

	for {
		select {
		case <-stopChan:
			return nil
		default:
			line, err := reader.ReadBytes('\n')
			if err != nil {
				return fmt.Errorf("stream ended or error occurred: %v", err)
			}

			var result Result
			err = json.Unmarshal(line, &result) 
			if err != nil {
				log.Printf("Error unmarshaling JSON: %v", err)
				continue
			}

			mu.Lock()
			for _, det := range result.Data {
				if det.Label == "Unknown" {
					continue
				} 
				(*detections_queue)[det.Label] = det   // overwrites older detection
			}
			mu.Unlock()
		}
	}
}
