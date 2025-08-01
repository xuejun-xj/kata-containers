/*
Cloud Hypervisor API

Local HTTP based API for managing and inspecting a cloud-hypervisor virtual machine.

API version: 0.3.0
*/

// Code generated by OpenAPI Generator (https://openapi-generator.tech); DO NOT EDIT.

package openapi

import (
	"encoding/json"
)

// NetConfig struct for NetConfig
type NetConfig struct {
	Tap *string `json:"tap,omitempty"`
	// IPv4 or IPv6 address
	Ip *string `json:"ip,omitempty"`
	// Must be a valid IPv4 netmask if ip is an IPv4 address or a valid IPv6 netmask if ip is an IPv6 address.
	Mask              *string            `json:"mask,omitempty"`
	Mac               *string            `json:"mac,omitempty"`
	HostMac           *string            `json:"host_mac,omitempty"`
	Mtu               *int32             `json:"mtu,omitempty"`
	Iommu             *bool              `json:"iommu,omitempty"`
	NumQueues         *int32             `json:"num_queues,omitempty"`
	QueueSize         *int32             `json:"queue_size,omitempty"`
	VhostUser         *bool              `json:"vhost_user,omitempty"`
	VhostSocket       *string            `json:"vhost_socket,omitempty"`
	VhostMode         *string            `json:"vhost_mode,omitempty"`
	Id                *string            `json:"id,omitempty"`
	PciSegment        *int32             `json:"pci_segment,omitempty"`
	RateLimiterConfig *RateLimiterConfig `json:"rate_limiter_config,omitempty"`
}

// NewNetConfig instantiates a new NetConfig object
// This constructor will assign default values to properties that have it defined,
// and makes sure properties required by API are set, but the set of arguments
// will change when the set of required properties is changed
func NewNetConfig() *NetConfig {
	this := NetConfig{}
	var ip string = "192.168.249.1"
	this.Ip = &ip
	var mask string = "255.255.255.0"
	this.Mask = &mask
	var iommu bool = false
	this.Iommu = &iommu
	var numQueues int32 = 2
	this.NumQueues = &numQueues
	var queueSize int32 = 256
	this.QueueSize = &queueSize
	var vhostUser bool = false
	this.VhostUser = &vhostUser
	var vhostMode string = "Client"
	this.VhostMode = &vhostMode
	return &this
}

// NewNetConfigWithDefaults instantiates a new NetConfig object
// This constructor will only assign default values to properties that have it defined,
// but it doesn't guarantee that properties required by API are set
func NewNetConfigWithDefaults() *NetConfig {
	this := NetConfig{}
	var ip string = "192.168.249.1"
	this.Ip = &ip
	var mask string = "255.255.255.0"
	this.Mask = &mask
	var iommu bool = false
	this.Iommu = &iommu
	var numQueues int32 = 2
	this.NumQueues = &numQueues
	var queueSize int32 = 256
	this.QueueSize = &queueSize
	var vhostUser bool = false
	this.VhostUser = &vhostUser
	var vhostMode string = "Client"
	this.VhostMode = &vhostMode
	return &this
}

// GetTap returns the Tap field value if set, zero value otherwise.
func (o *NetConfig) GetTap() string {
	if o == nil || o.Tap == nil {
		var ret string
		return ret
	}
	return *o.Tap
}

// GetTapOk returns a tuple with the Tap field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetTapOk() (*string, bool) {
	if o == nil || o.Tap == nil {
		return nil, false
	}
	return o.Tap, true
}

// HasTap returns a boolean if a field has been set.
func (o *NetConfig) HasTap() bool {
	if o != nil && o.Tap != nil {
		return true
	}

	return false
}

// SetTap gets a reference to the given string and assigns it to the Tap field.
func (o *NetConfig) SetTap(v string) {
	o.Tap = &v
}

// GetIp returns the Ip field value if set, zero value otherwise.
func (o *NetConfig) GetIp() string {
	if o == nil || o.Ip == nil {
		var ret string
		return ret
	}
	return *o.Ip
}

// GetIpOk returns a tuple with the Ip field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetIpOk() (*string, bool) {
	if o == nil || o.Ip == nil {
		return nil, false
	}
	return o.Ip, true
}

// HasIp returns a boolean if a field has been set.
func (o *NetConfig) HasIp() bool {
	if o != nil && o.Ip != nil {
		return true
	}

	return false
}

// SetIp gets a reference to the given string and assigns it to the Ip field.
func (o *NetConfig) SetIp(v string) {
	o.Ip = &v
}

// GetMask returns the Mask field value if set, zero value otherwise.
func (o *NetConfig) GetMask() string {
	if o == nil || o.Mask == nil {
		var ret string
		return ret
	}
	return *o.Mask
}

// GetMaskOk returns a tuple with the Mask field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetMaskOk() (*string, bool) {
	if o == nil || o.Mask == nil {
		return nil, false
	}
	return o.Mask, true
}

// HasMask returns a boolean if a field has been set.
func (o *NetConfig) HasMask() bool {
	if o != nil && o.Mask != nil {
		return true
	}

	return false
}

// SetMask gets a reference to the given string and assigns it to the Mask field.
func (o *NetConfig) SetMask(v string) {
	o.Mask = &v
}

// GetMac returns the Mac field value if set, zero value otherwise.
func (o *NetConfig) GetMac() string {
	if o == nil || o.Mac == nil {
		var ret string
		return ret
	}
	return *o.Mac
}

// GetMacOk returns a tuple with the Mac field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetMacOk() (*string, bool) {
	if o == nil || o.Mac == nil {
		return nil, false
	}
	return o.Mac, true
}

// HasMac returns a boolean if a field has been set.
func (o *NetConfig) HasMac() bool {
	if o != nil && o.Mac != nil {
		return true
	}

	return false
}

// SetMac gets a reference to the given string and assigns it to the Mac field.
func (o *NetConfig) SetMac(v string) {
	o.Mac = &v
}

// GetHostMac returns the HostMac field value if set, zero value otherwise.
func (o *NetConfig) GetHostMac() string {
	if o == nil || o.HostMac == nil {
		var ret string
		return ret
	}
	return *o.HostMac
}

// GetHostMacOk returns a tuple with the HostMac field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetHostMacOk() (*string, bool) {
	if o == nil || o.HostMac == nil {
		return nil, false
	}
	return o.HostMac, true
}

// HasHostMac returns a boolean if a field has been set.
func (o *NetConfig) HasHostMac() bool {
	if o != nil && o.HostMac != nil {
		return true
	}

	return false
}

// SetHostMac gets a reference to the given string and assigns it to the HostMac field.
func (o *NetConfig) SetHostMac(v string) {
	o.HostMac = &v
}

// GetMtu returns the Mtu field value if set, zero value otherwise.
func (o *NetConfig) GetMtu() int32 {
	if o == nil || o.Mtu == nil {
		var ret int32
		return ret
	}
	return *o.Mtu
}

// GetMtuOk returns a tuple with the Mtu field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetMtuOk() (*int32, bool) {
	if o == nil || o.Mtu == nil {
		return nil, false
	}
	return o.Mtu, true
}

// HasMtu returns a boolean if a field has been set.
func (o *NetConfig) HasMtu() bool {
	if o != nil && o.Mtu != nil {
		return true
	}

	return false
}

// SetMtu gets a reference to the given int32 and assigns it to the Mtu field.
func (o *NetConfig) SetMtu(v int32) {
	o.Mtu = &v
}

// GetIommu returns the Iommu field value if set, zero value otherwise.
func (o *NetConfig) GetIommu() bool {
	if o == nil || o.Iommu == nil {
		var ret bool
		return ret
	}
	return *o.Iommu
}

// GetIommuOk returns a tuple with the Iommu field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetIommuOk() (*bool, bool) {
	if o == nil || o.Iommu == nil {
		return nil, false
	}
	return o.Iommu, true
}

// HasIommu returns a boolean if a field has been set.
func (o *NetConfig) HasIommu() bool {
	if o != nil && o.Iommu != nil {
		return true
	}

	return false
}

// SetIommu gets a reference to the given bool and assigns it to the Iommu field.
func (o *NetConfig) SetIommu(v bool) {
	o.Iommu = &v
}

// GetNumQueues returns the NumQueues field value if set, zero value otherwise.
func (o *NetConfig) GetNumQueues() int32 {
	if o == nil || o.NumQueues == nil {
		var ret int32
		return ret
	}
	return *o.NumQueues
}

// GetNumQueuesOk returns a tuple with the NumQueues field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetNumQueuesOk() (*int32, bool) {
	if o == nil || o.NumQueues == nil {
		return nil, false
	}
	return o.NumQueues, true
}

// HasNumQueues returns a boolean if a field has been set.
func (o *NetConfig) HasNumQueues() bool {
	if o != nil && o.NumQueues != nil {
		return true
	}

	return false
}

// SetNumQueues gets a reference to the given int32 and assigns it to the NumQueues field.
func (o *NetConfig) SetNumQueues(v int32) {
	o.NumQueues = &v
}

// GetQueueSize returns the QueueSize field value if set, zero value otherwise.
func (o *NetConfig) GetQueueSize() int32 {
	if o == nil || o.QueueSize == nil {
		var ret int32
		return ret
	}
	return *o.QueueSize
}

// GetQueueSizeOk returns a tuple with the QueueSize field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetQueueSizeOk() (*int32, bool) {
	if o == nil || o.QueueSize == nil {
		return nil, false
	}
	return o.QueueSize, true
}

// HasQueueSize returns a boolean if a field has been set.
func (o *NetConfig) HasQueueSize() bool {
	if o != nil && o.QueueSize != nil {
		return true
	}

	return false
}

// SetQueueSize gets a reference to the given int32 and assigns it to the QueueSize field.
func (o *NetConfig) SetQueueSize(v int32) {
	o.QueueSize = &v
}

// GetVhostUser returns the VhostUser field value if set, zero value otherwise.
func (o *NetConfig) GetVhostUser() bool {
	if o == nil || o.VhostUser == nil {
		var ret bool
		return ret
	}
	return *o.VhostUser
}

// GetVhostUserOk returns a tuple with the VhostUser field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetVhostUserOk() (*bool, bool) {
	if o == nil || o.VhostUser == nil {
		return nil, false
	}
	return o.VhostUser, true
}

// HasVhostUser returns a boolean if a field has been set.
func (o *NetConfig) HasVhostUser() bool {
	if o != nil && o.VhostUser != nil {
		return true
	}

	return false
}

// SetVhostUser gets a reference to the given bool and assigns it to the VhostUser field.
func (o *NetConfig) SetVhostUser(v bool) {
	o.VhostUser = &v
}

// GetVhostSocket returns the VhostSocket field value if set, zero value otherwise.
func (o *NetConfig) GetVhostSocket() string {
	if o == nil || o.VhostSocket == nil {
		var ret string
		return ret
	}
	return *o.VhostSocket
}

// GetVhostSocketOk returns a tuple with the VhostSocket field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetVhostSocketOk() (*string, bool) {
	if o == nil || o.VhostSocket == nil {
		return nil, false
	}
	return o.VhostSocket, true
}

// HasVhostSocket returns a boolean if a field has been set.
func (o *NetConfig) HasVhostSocket() bool {
	if o != nil && o.VhostSocket != nil {
		return true
	}

	return false
}

// SetVhostSocket gets a reference to the given string and assigns it to the VhostSocket field.
func (o *NetConfig) SetVhostSocket(v string) {
	o.VhostSocket = &v
}

// GetVhostMode returns the VhostMode field value if set, zero value otherwise.
func (o *NetConfig) GetVhostMode() string {
	if o == nil || o.VhostMode == nil {
		var ret string
		return ret
	}
	return *o.VhostMode
}

// GetVhostModeOk returns a tuple with the VhostMode field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetVhostModeOk() (*string, bool) {
	if o == nil || o.VhostMode == nil {
		return nil, false
	}
	return o.VhostMode, true
}

// HasVhostMode returns a boolean if a field has been set.
func (o *NetConfig) HasVhostMode() bool {
	if o != nil && o.VhostMode != nil {
		return true
	}

	return false
}

// SetVhostMode gets a reference to the given string and assigns it to the VhostMode field.
func (o *NetConfig) SetVhostMode(v string) {
	o.VhostMode = &v
}

// GetId returns the Id field value if set, zero value otherwise.
func (o *NetConfig) GetId() string {
	if o == nil || o.Id == nil {
		var ret string
		return ret
	}
	return *o.Id
}

// GetIdOk returns a tuple with the Id field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetIdOk() (*string, bool) {
	if o == nil || o.Id == nil {
		return nil, false
	}
	return o.Id, true
}

// HasId returns a boolean if a field has been set.
func (o *NetConfig) HasId() bool {
	if o != nil && o.Id != nil {
		return true
	}

	return false
}

// SetId gets a reference to the given string and assigns it to the Id field.
func (o *NetConfig) SetId(v string) {
	o.Id = &v
}

// GetPciSegment returns the PciSegment field value if set, zero value otherwise.
func (o *NetConfig) GetPciSegment() int32 {
	if o == nil || o.PciSegment == nil {
		var ret int32
		return ret
	}
	return *o.PciSegment
}

// GetPciSegmentOk returns a tuple with the PciSegment field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetPciSegmentOk() (*int32, bool) {
	if o == nil || o.PciSegment == nil {
		return nil, false
	}
	return o.PciSegment, true
}

// HasPciSegment returns a boolean if a field has been set.
func (o *NetConfig) HasPciSegment() bool {
	if o != nil && o.PciSegment != nil {
		return true
	}

	return false
}

// SetPciSegment gets a reference to the given int32 and assigns it to the PciSegment field.
func (o *NetConfig) SetPciSegment(v int32) {
	o.PciSegment = &v
}

// GetRateLimiterConfig returns the RateLimiterConfig field value if set, zero value otherwise.
func (o *NetConfig) GetRateLimiterConfig() RateLimiterConfig {
	if o == nil || o.RateLimiterConfig == nil {
		var ret RateLimiterConfig
		return ret
	}
	return *o.RateLimiterConfig
}

// GetRateLimiterConfigOk returns a tuple with the RateLimiterConfig field value if set, nil otherwise
// and a boolean to check if the value has been set.
func (o *NetConfig) GetRateLimiterConfigOk() (*RateLimiterConfig, bool) {
	if o == nil || o.RateLimiterConfig == nil {
		return nil, false
	}
	return o.RateLimiterConfig, true
}

// HasRateLimiterConfig returns a boolean if a field has been set.
func (o *NetConfig) HasRateLimiterConfig() bool {
	if o != nil && o.RateLimiterConfig != nil {
		return true
	}

	return false
}

// SetRateLimiterConfig gets a reference to the given RateLimiterConfig and assigns it to the RateLimiterConfig field.
func (o *NetConfig) SetRateLimiterConfig(v RateLimiterConfig) {
	o.RateLimiterConfig = &v
}

func (o NetConfig) MarshalJSON() ([]byte, error) {
	toSerialize := map[string]interface{}{}
	if o.Tap != nil {
		toSerialize["tap"] = o.Tap
	}
	if o.Ip != nil {
		toSerialize["ip"] = o.Ip
	}
	if o.Mask != nil {
		toSerialize["mask"] = o.Mask
	}
	if o.Mac != nil {
		toSerialize["mac"] = o.Mac
	}
	if o.HostMac != nil {
		toSerialize["host_mac"] = o.HostMac
	}
	if o.Mtu != nil {
		toSerialize["mtu"] = o.Mtu
	}
	if o.Iommu != nil {
		toSerialize["iommu"] = o.Iommu
	}
	if o.NumQueues != nil {
		toSerialize["num_queues"] = o.NumQueues
	}
	if o.QueueSize != nil {
		toSerialize["queue_size"] = o.QueueSize
	}
	if o.VhostUser != nil {
		toSerialize["vhost_user"] = o.VhostUser
	}
	if o.VhostSocket != nil {
		toSerialize["vhost_socket"] = o.VhostSocket
	}
	if o.VhostMode != nil {
		toSerialize["vhost_mode"] = o.VhostMode
	}
	if o.Id != nil {
		toSerialize["id"] = o.Id
	}
	if o.PciSegment != nil {
		toSerialize["pci_segment"] = o.PciSegment
	}
	if o.RateLimiterConfig != nil {
		toSerialize["rate_limiter_config"] = o.RateLimiterConfig
	}
	return json.Marshal(toSerialize)
}

type NullableNetConfig struct {
	value *NetConfig
	isSet bool
}

func (v NullableNetConfig) Get() *NetConfig {
	return v.value
}

func (v *NullableNetConfig) Set(val *NetConfig) {
	v.value = val
	v.isSet = true
}

func (v NullableNetConfig) IsSet() bool {
	return v.isSet
}

func (v *NullableNetConfig) Unset() {
	v.value = nil
	v.isSet = false
}

func NewNullableNetConfig(val *NetConfig) *NullableNetConfig {
	return &NullableNetConfig{value: val, isSet: true}
}

func (v NullableNetConfig) MarshalJSON() ([]byte, error) {
	return json.Marshal(v.value)
}

func (v *NullableNetConfig) UnmarshalJSON(src []byte) error {
	v.isSet = true
	return json.Unmarshal(src, &v.value)
}
